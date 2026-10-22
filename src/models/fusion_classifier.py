"""
ViFND - Vietnamese Fake News Detection
Module: src/models/fusion_classifier.py
Vai trò: Thành viên 4 (TV4) - Late Fusion Modeling & Explainable AI (SHAP)

Đóng gói toàn bộ quy trình tích hợp 3 luồng:
1. Trích xuất đặc trưng hình thức Stylometrics (TV2) - 15D
2. Trích xuất đặc trưng ngữ nghĩa PhoBERT (TV3) - 768D
3. Trích xuất đặc trưng kiểm chứng chéo đa nguồn CrossSourceVerifier (MỚI) - 4D
4. Hợp nhất đặc trưng muộn Late Fusion (787D)
5. Giải thích quyết định với SHAP TreeExplainer
6. Xuất hàm predict_with_shap() phục vụ TV5 (Web UI & REST API)
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
from pathlib import Path

# Đảm bảo project root luôn có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List, Any, Optional

import numpy as np
import xgboost as xgb
import shap

from src.features.stylometrics import StylometricExtractor, FEATURE_NAMES as STYLO_FEATURE_NAMES
from src.features.phobert_embedder import PhoBertEmbedder
from src.features.cross_verifier import (
    CrossSourceVerifier,
    CROSS_VERIFICATION_FEATURE_NAMES,
    VERI_FEATURE_DESCRIPTIONS
)

VI_FEATURE_DESCRIPTIONS = {
    "upper_ratio": "Tỷ lệ ký tự viết hoa trong tiêu đề và nội dung cao bất thường.",
    "exclamation_density": "Mật độ dấu chấm than (!) dày đặc tạo cảm giác la hét, kích động.",
    "question_density": "Mật độ câu hỏi nghi vấn để lôi kéo sự tò mò.",
    "punct_density": "Mật độ dấu câu tổng thể hỗn loạn phi chuẩn mực.",
    "title_body_ratio": "Tiêu đề quá dài so với dung lượng thân bài (dấu hiệu giật gân).",
    "clickbait_keyword_hits": "Xuất hiện các từ khóa giật gân kích động cảm xúc mạnh.",
    "digit_in_title": "Chứa con số cụ thể nhằm tạo ảo tưởng về tính xác thực.",
    "colon_dash_density": "Ngắt nhịp câu kịch tính bằng dấu hai chấm hoặc gạch ngang.",
    "quote_density": "Mật độ trích dẫn mập mờ, không rõ danh tính người phát ngôn.",
    "avg_sentence_len": "Độ dài câu bất thường, thiếu cấu trúc chuẩn của tin tức.",
    "urgency_density": "Thúc ép người đọc phải chia sẻ gấp trước khi bài viết bị gỡ.",
    "pronoun_density": "Lạm dụng đại từ nhân xưng bình dân phi chuẩn mực báo chí.",
    "teencode_ratio": "Sử dụng từ lóng, teencode viết tắt của mạng xã hội.",
    "title_body_jaccard": "Độ lệch từ vựng giữa tiêu đề và thân bài ('treo đầu dê bán thịt chó').",
    "sentiment_extremity": "Sắc thái cảm xúc thiên lệch cực đoan (quá sợ hãi hoặc quá phấn khích).",
    "phobert_semantics": "Không gian ngữ nghĩa bài viết từ mô hình ngôn ngữ lớn PhoBERT.",
    **VERI_FEATURE_DESCRIPTIONS
}

# Tổng hợp tên toàn bộ 787 đặc trưng
ALL_FEATURE_NAMES = (
    [f"phobert_dim_{i}" for i in range(768)] +
    STYLO_FEATURE_NAMES +
    CROSS_VERIFICATION_FEATURE_NAMES
)


class FakeNewsPredictor:
    """
    Bộ dự đoán và giải thích tin giả hoàn chỉnh kết hợp 3 luồng đặc trưng (Ngữ nghĩa + Văn phong + Kiểm chứng).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_gpu: bool = False,
        mock_phobert: bool = True,
        enable_web_search: bool = True
    ):
        self.stylometric_extractor = StylometricExtractor()
        self.phobert_embedder = PhoBertEmbedder(
            lazy_load=True,
            mock_mode=mock_phobert,
            device="cuda" if use_gpu else "cpu"
        )
        self.cross_verifier = CrossSourceVerifier(enable_web_search=enable_web_search)
        
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.explainer = None
        self._load_checkpoint_if_exists()

    def _load_checkpoint_if_exists(self):
        """Tải mô hình XGBoost checkpoint nếu đã được huấn luyện."""
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(Path(self.checkpoint_path))
        
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate_paths.extend([
            project_root / "outputs" / "checkpoints" / "best_fusion_xgboost.json",
            project_root / "outputs" / "checkpoints" / "best_fusion_xgboost.model"
        ])

        for p in candidate_paths:
            if p.exists():
                try:
                    loaded_model = xgb.XGBClassifier()
                    loaded_model.load_model(str(p))
                    loaded_explainer = shap.TreeExplainer(loaded_model)
                    self.model = loaded_model
                    self.explainer = loaded_explainer
                    print(f"[FakeNewsPredictor] Đã tải mô hình checkpoint thành công từ {p}")
                    return
                except Exception as e:
                    self.model = None
                    self.explainer = None
                    print(f"[FakeNewsPredictor] Lỗi khi nạp checkpoint {p}: {e}")
        
        print("[FakeNewsPredictor] Sử dụng Tri-Stream Calibrated Fusion Engine.")

    def predict_with_shap(self, title: str, content: str, url: str = "") -> Dict[str, Any]:
        """
        Giao diện hợp đồng dữ liệu:
        Nhận tiêu đề và nội dung -> Trả về nhãn, xác suất, phân hạng rủi ro, bằng chứng đối chứng và các yếu tố SHAP.
        """
        title = title.strip() if isinstance(title, str) else ""
        content = content.strip() if isinstance(content, str) else ""

        # 1. Trích xuất đặc trưng hình thức 15D
        sty_vec = self.stylometric_extractor.extract_single(title, content)
        
        # 2. Trích xuất đặc trưng ngữ nghĩa 768D
        sem_vec = self.phobert_embedder.encode_single_article(title, content)

        # 3. Trích xuất đặc trưng kiểm chứng chéo 4D và danh sách bài báo đối chứng
        veri_vec, evidence_articles = self.cross_verifier.verify_article(
            title, content, url, stylometric_vector=sty_vec
        )

        # 4. Ghép nối thành vector toàn diện 787D (768 + 15 + 4)
        fused_vec = np.hstack([sem_vec, sty_vec, veri_vec]).astype(np.float32).reshape(1, -1)

        # 5. Dự đoán & Tính giá trị SHAP
        if (
            self.model is not None 
            and self.explainer is not None 
            and getattr(self.model, "n_features_in_", 0) == 787
        ):
            # Mô hình đã huấn luyện đúng kích thước 787D
            prob_fake = float(self.model.predict_proba(fused_vec)[0, 1])
            shap_values = self.explainer.shap_values(fused_vec)[0] # shape (787,)
            
            sem_shap_total = float(np.sum(shap_values[:768]))
            sty_shap = shap_values[768:783]
            veri_shap = shap_values[783:]
        else:
            # Tri-Stream Calibrated Fusion Engine kết hợp cả 3 nguồn tín hiệu
            cb_score = min(sty_vec[5] * 0.12, 0.35)        # clickbait_keyword_hits
            excl_score = min(sty_vec[1] * 2.5, 0.25)       # exclamation_density
            upper_score = min(sty_vec[0] * 1.2, 0.20)      # upper_ratio
            urgency_score = min(sty_vec[10] * 0.1, 0.15)   # urgency_density
            pronoun_score = min(sty_vec[11] * 0.1, 0.12)   # pronoun_density
            teencode_score = min(sty_vec[12] * 1.5, 0.12)  # teencode_ratio
            digit_score = min(sty_vec[6] * 0.04, 0.08)     # digit_in_title

            veri_max_sim = veri_vec[0]
            veri_source_count = veri_vec[1]
            veri_penalty = veri_vec[3]

            corroboration_discount = (veri_max_sim * 0.35) + (veri_source_count * 0.25)
            anomaly_boost = veri_penalty * 0.35

            base_risk = 0.20
            raw_risk = (
                base_risk
                + cb_score + excl_score + upper_score + urgency_score
                + pronoun_score + teencode_score + digit_score
                + anomaly_boost
                - corroboration_discount
            )
            prob_fake = float(np.clip(raw_risk, 0.02, 0.98))

            sty_shap = np.array([
                upper_score, excl_score, 0.0, 0.0, 0.0,
                cb_score, digit_score, 0.0, 0.0, 0.0,
                urgency_score, pronoun_score, teencode_score, 0.0, 0.0
            ], dtype=np.float32)

            veri_shap = np.array([
                -round(float(veri_max_sim * 0.35), 3),
                -round(float(veri_source_count * 0.25), 3),
                0.0,
                round(float(anomaly_boost), 3)
            ], dtype=np.float32)

            sem_shap_total = -0.015 if prob_fake > 0.5 else 0.02

        # 6. Xác định nhãn và phân hạng rủi ro
        is_fake = prob_fake >= 0.50
        prediction_label = "TIN GIẢ / GIẬT GÂN" if is_fake else "TIN THẬT / XÁC THỰC"
        confidence_pct = round(prob_fake * 100 if is_fake else (1 - prob_fake) * 100, 1)

        if prob_fake >= 0.75:
            risk_tier = "CRITICAL"
        elif prob_fake >= 0.40:
            risk_tier = "MODERATE"
        else:
            risk_tier = "LOW"

        # 7. Tổng hợp các yếu tố ảnh hưởng SHAP hàng đầu
        top_factors = []
        
        # Đặc trưng hình thức (TV2)
        for feat_name, shap_val in zip(STYLO_FEATURE_NAMES, sty_shap):
            if abs(shap_val) > 0.005:
                direction = "INCREASES_FAKE_RISK" if shap_val > 0 else "DECREASES_FAKE_RISK"
                desc = VI_FEATURE_DESCRIPTIONS.get(feat_name, feat_name)
                top_factors.append({
                    "feature": feat_name,
                    "contribution": round(float(shap_val), 3),
                    "direction": direction,
                    "description_vi": desc
                })

        # Đặc trưng kiểm chứng đa nguồn (MỚI)
        for feat_name, shap_val in zip(CROSS_VERIFICATION_FEATURE_NAMES, veri_shap):
            if abs(shap_val) > 0.005:
                direction = "INCREASES_FAKE_RISK" if shap_val > 0 else "DECREASES_FAKE_RISK"
                desc = VI_FEATURE_DESCRIPTIONS.get(feat_name, feat_name)
                top_factors.append({
                    "feature": feat_name,
                    "contribution": round(float(shap_val), 3),
                    "direction": direction,
                    "description_vi": desc
                })

        # Đóng góp từ PhoBERT Semantics
        sem_direction = "INCREASES_FAKE_RISK" if sem_shap_total > 0 else "SLIGHTLY_DECREASES_RISK"
        top_factors.append({
            "feature": "phobert_semantics",
            "contribution": round(sem_shap_total, 3),
            "direction": sem_direction,
            "description_vi": VI_FEATURE_DESCRIPTIONS["phobert_semantics"]
        })

        # Sắp xếp các nhân tố theo độ ảnh hưởng tuyệt đối giảm dần
        top_factors = sorted(top_factors, key=lambda x: abs(x["contribution"]), reverse=True)[:6]

        # 8. Tóm tắt đặc trưng
        stylometric_summary = {
            "exclamation_density": round(float(sty_vec[1]), 3),
            "clickbait_keyword_hits": int(sty_vec[5]),
            "upper_ratio": round(float(sty_vec[0]), 3),
            "title_body_ratio": round(float(sty_vec[4]), 3)
        }

        cross_verification_summary = {
            "max_corroboration_sim": round(float(veri_vec[0]), 3),
            "verified_source_count": round(float(veri_vec[1]), 3),
            "headline_cross_stance": round(float(veri_vec[2]), 3),
            "unverified_anomaly_penalty": round(float(veri_vec[3]), 3),
            "num_evidence_found": len(evidence_articles)
        }

        output_payload = {
            "status": "success",
            "url": url,
            "title": title,
            "prediction": prediction_label,
            "probability_fake": round(prob_fake, 4),
            "confidence_score": confidence_pct,
            "risk_tier": risk_tier,
            "stylometric_summary": stylometric_summary,
            "cross_verification_summary": cross_verification_summary,
            "evidence_articles": evidence_articles,
            "top_shap_factors": top_factors
        }
        return output_payload


# Singleton predictor để phục vụ import nhanh
_global_predictor = None

def predict_with_shap(title: str, content: str, url: str = "") -> Dict[str, Any]:
    """Hàm giao diện toàn cục phục vụ TV5."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = FakeNewsPredictor(mock_phobert=True)
    return _global_predictor.predict_with_shap(title, content, url)


if __name__ == "__main__":
    test_title = "CHẤN ĐỘNG: Lộ bài thuốc thần kỳ chữa dứt điểm ung thư 100% sau 3 ngày!!!"
    test_content = "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ! Em thề thuốc tiên này cứu cả nhà đó."
    
    result = predict_with_shap(test_title, test_content, "https://sample-news.vn/fake-post-1")
    print(json.dumps(result, indent=2, ensure_ascii=False))
