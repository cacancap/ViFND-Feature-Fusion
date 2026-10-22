"""
ViFND - Vietnamese Fake News Detection
Module: src/features/cross_verifier.py
Tầng: Cross-Source Corroboration Engine (Tầng Kiểm chứng Chéo Đa Nguồn - Hybrid)

Chức năng:
1. Kiểm chứng chéo sự kiện với Kho báo chí chính thống cục bộ (Local Index: BM25/TF-IDF trên tập nhãn 0).
2. Fallback tìm kiếm đối chứng thời gian thực (Google News RSS / Web News) khi cần hoặc khi demo online.
3. Trích xuất 4 đặc trưng kiểm chứng toán học cho Late Fusion:
   - veri_max_corroboration_sim: Độ tương đồng cao nhất với bài báo chính thống (0.0 - 1.0)
   - veri_verified_source_count: Số lượng nguồn báo chí lớn cùng đưa tin (chuẩn hóa 0.0 - 1.0)
   - veri_headline_cross_stance: Độ nhất quán giữa tiêu đề truy vấn và tiêu đề chính thống (0.0 - 1.0)
   - veri_unverified_anomaly_penalty: Điểm phạt khi tin có dấu hiệu giật gân/bất thường nhưng 0 nguồn chính thống nào xác nhận
4. Lọc thực thể địa lý (Entity Mismatch Guard) ngăn chặn việc nhầm lẫn sự kiện nước ngoài với sự kiện trong nước.
5. Trả về danh sách bài báo đối chứng (Evidence Articles) để hiển thị trực quan lên Web UI.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import re
import xml.etree.ElementTree as ET
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

CROSS_VERIFICATION_FEATURE_NAMES = [
    "veri_max_corroboration_sim",
    "veri_verified_source_count",
    "veri_headline_cross_stance",
    "veri_unverified_anomaly_penalty"
]

VERI_FEATURE_DESCRIPTIONS = {
    "veri_max_corroboration_sim": "Mức độ tương đồng nội dung với các bài báo từ nguồn chính thống.",
    "veri_verified_source_count": "Số lượng cơ quan báo chí uy tín cùng đưa tin về sự kiện.",
    "veri_headline_cross_stance": "Độ nhất quán giữa tiêu đề truy vấn và thông tin từ báo chí chính thống.",
    "veri_unverified_anomaly_penalty": "Điểm phạt bất thường khi đưa tin nhưng không có nguồn chính thống nào xác nhận."
}

TRUSTED_DOMAINS = [
    "vnexpress.net", "tuoitre.vn", "vietnamnet.vn", "thanhnien.vn",
    "dantri.com.vn", "vtv.vn", "baochinhphu.vn", "nhandan.vn",
    "cafef.vn", "plo.vn", "sggp.org.vn", "chinhphu.vn", "tingia.gov.vn",
    "moet.gov.vn", "moh.gov.vn", "cand.com.vn", "qdnd.vn"
]

FOREIGN_ENTITIES = {
    "pháp", "nga", "mỹ", "hoa kỳ", "trung quốc", "bắc kinh", "nhật", "nhật bản", 
    "hàn", "hàn quốc", "anh", "luân đôn", "đức", "úc", "ukraine", "israel", "iran"
}


class CrossSourceVerifier:
    """
    Bộ kiểm chứng chéo đa nguồn Hybrid kết hợp kho cục bộ và tìm kiếm đối chứng.
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        enable_web_search: bool = True,
        web_search_timeout: float = 3.5
    ):
        self.enable_web_search = enable_web_search
        self.web_search_timeout = web_search_timeout
        self.corpus_path = corpus_path
        self.corpus_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_tfidf = None
        
        self._initialize_local_index()

    def _initialize_local_index(self):
        """Khởi tạo kho ngữ liệu chính thống và bộ lập chỉ mục TF-IDF."""
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate_paths = []
        if self.corpus_path:
            candidate_paths.append(Path(self.corpus_path))
        
        candidate_paths.extend([
            project_root / "data" / "processed" / "corpus_dedup.parquet",
            project_root / "data" / "fixtures" / "sample_corpus.parquet",
            project_root / "data" / "fixtures" / "sample_corpus.csv"
        ])

        target_file = None
        for p in candidate_paths:
            if p.exists():
                target_file = p
                break

        if target_file is not None:
            try:
                if str(target_file).endswith(".parquet"):
                    df = pd.read_parquet(target_file)
                else:
                    df = pd.read_csv(target_file)
                
                # Lọc các bài báo chính thống (label == 0)
                if "label" in df.columns:
                    verified_df = df[df["label"] == 0].copy()
                else:
                    verified_df = df.copy()

                if len(verified_df) > 0:
                    self.corpus_df = verified_df.reset_index(drop=True)
                    combined_texts = (
                        self.corpus_df["title"].fillna("") + " " +
                        self.corpus_df["content"].fillna("")
                    ).tolist()

                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=8000,
                        ngram_range=(1, 2)
                    )
                    self.corpus_tfidf = self.tfidf_vectorizer.fit_transform(combined_texts)
            except Exception as e:
                print(f"[CrossSourceVerifier] Lỗi khi lập chỉ mục kho cục bộ: {e}")

    def _clean_query_keywords(self, title: str) -> str:
        """Trích xuất từ khóa trọng tâm từ tiêu đề bài viết để truy vấn."""
        clean = re.sub(r'^(CHẤN ĐỘNG|CẢNH BÁO|BÍ MẬT|BẬT MÍ|TIN NÓNG|NÓNG|HOT|HÉ LỘ)[\s\:\!\?]+', '', title, flags=re.IGNORECASE)
        clean = re.sub(r'[!\?\"\'\(\)\[\]\{\}\#\$\%]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        stop_words = {"và", "các", "những", "cho", "của", "tại", "từ", "năm", "về", "được", "ra", "vào", "qua"}
        words = [w for w in clean.split() if w.lower() not in stop_words]
        if not words:
            words = clean.split()
            
        if len(words) > 8:
            clean = " ".join(words[:8])
        else:
            clean = " ".join(words)
        return clean

    def _has_foreign_entity_conflict(self, query: str, candidate_title: str) -> bool:
        """Kiểm tra xem ứng viên tìm thấy có bị xung đột thực thể địa lý nước ngoài hay không."""
        q_lower = query.lower()
        cand_lower = candidate_title.lower()
        
        query_has_foreign = any(re.search(rf'\b{fe}\b', q_lower) for fe in FOREIGN_ENTITIES)
        cand_has_foreign = any(re.search(rf'\b{fe}\b', cand_lower) for fe in FOREIGN_ENTITIES)
        
        # Nếu bài viết hỏi về trong nước mà kết quả trả về là sự kiện ở nước ngoài -> Xung đột!
        if cand_has_foreign and not query_has_foreign:
            return True
        return False

    def _query_local_corpus(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Truy vấn kho lưu trữ báo chính thống cục bộ."""
        if self.corpus_df is None or self.tfidf_vectorizer is None or self.corpus_tfidf is None:
            return []

        query_text = f"{title} {content[:400]}"
        try:
            q_vec = self.tfidf_vectorizer.transform([query_text])
            sims = cosine_similarity(q_vec, self.corpus_tfidf).flatten()

            top_indices = np.argsort(sims)[::-1][:5]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                row = self.corpus_df.iloc[idx]
                cand_title = str(row.get("title", ""))
                
                # Áp dụng bộ lọc xung đột thực thể
                if self._has_foreign_entity_conflict(title, cand_title):
                    score *= 0.15
                    
                if score > 0.10:
                    results.append({
                        "title": cand_title,
                        "source": str(row.get("source", "Báo chính thống")),
                        "url": str(row.get("url", "")),
                        "similarity": round(score, 3)
                    })
            return results
        except Exception:
            return []

    def _query_google_news_rss(self, title: str) -> List[Dict[str, Any]]:
        """Truy vấn tin tức xác thực thời gian thực qua Google News RSS tiếng Việt."""
        if not self.enable_web_search:
            return []

        query = self._clean_query_keywords(title)
        if not query:
            return []

        try:
            encoded_q = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=vi&gl=VN&ceid=VN:vi"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

            resp = requests.get(rss_url, headers=headers, timeout=self.web_search_timeout)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            items = root.findall(".//item")[:5]

            results = []
            for item in items:
                item_title = item.find("title").text if item.find("title") is not None else ""
                item_link = item.find("link").text if item.find("link") is not None else ""
                item_source = "Báo chí trực tuyến"
                if " - " in item_title:
                    parts = item_title.rsplit(" - ", 1)
                    item_title = parts[0].strip()
                    item_source = parts[1].strip()

                q_tokens = set(re.findall(r'\w+', query.lower()))
                cand_tokens = set(re.findall(r'\w+', item_title.lower()))
                overlap = len(q_tokens & cand_tokens)
                jaccard = overlap / max(len(q_tokens | cand_tokens), 1)
                recall_sim = overlap / max(len(q_tokens), 1)
                
                sim_score = 0.65 * recall_sim + 0.35 * jaccard

                # Nếu bị xung đột thực thể địa lý nước ngoài (ví dụ Dân trí đưa tin Pháp giới nghiêm) -> Phạt nặng sim_score
                if self._has_foreign_entity_conflict(title, item_title):
                    sim_score *= 0.15

                results.append({
                    "title": item_title,
                    "source": item_source,
                    "url": item_link,
                    "similarity": round(float(sim_score), 3)
                })
            return results
        except Exception:
            return []

    def verify_article(
        self,
        title: str,
        content: str,
        url: str = "",
        stylometric_vector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Trích xuất 4 đặc trưng kiểm chứng toán học và danh sách bằng chứng đối chứng.
        """
        title = title.strip() if isinstance(title, str) else ""
        content = content.strip() if isinstance(content, str) else ""

        evidence_list: List[Dict[str, Any]] = []

        # 1. Truy vấn kho cục bộ trước
        local_results = self._query_local_corpus(title, content)
        evidence_list.extend(local_results)

        # 2. Bổ sung truy vấn Google News RSS để tìm kiếm đối chứng diện rộng
        max_local_sim = max([r["similarity"] for r in local_results], default=0.0)
        if max_local_sim < 0.45 and self.enable_web_search:
            web_results = self._query_google_news_rss(title)
            evidence_list.extend(web_results)

        # Sắp xếp theo similarity giảm dần và lấy top 5
        evidence_list = sorted(evidence_list, key=lambda x: x["similarity"], reverse=True)[:5]

        # 3. Tính toán 4 đặc trưng kiểm chứng:
        if evidence_list:
            max_sim = float(evidence_list[0]["similarity"])
            top_evidence_title = evidence_list[0]["title"]
            q_set = set(re.findall(r'\w+', title.lower()))
            e_set = set(re.findall(r'\w+', top_evidence_title.lower()))
            headline_stance = float(len(q_set & e_set) / max(len(q_set | e_set), 1))

            # Nguồn uy tín xác nhận phải đạt độ tương đồng thực thể cao (>= 0.45)
            verified_sources = {
                r["source"].lower() for r in evidence_list
                if r["similarity"] >= 0.45 and r["source"]
            }
            source_count = float(min(len(verified_sources), 5)) / 5.0
        else:
            max_sim = 0.0
            source_count = 0.0
            headline_stance = 0.0

        # 4. Tính điểm phạt bất thường (Unverified Anomaly Penalty)
        sensational_cues = 0.0
        if stylometric_vector is not None and len(stylometric_vector) >= 6:
            upper_ratio = stylometric_vector[0]
            excl_density = stylometric_vector[1]
            clickbait_hits = stylometric_vector[5]
            sensational_cues = (upper_ratio * 1.0) + (min(excl_density, 0.04) * 1.5) + (min(clickbait_hits, 4) * 0.15)
        else:
            has_clickbait_words = bool(re.search(r'(chấn động|kinh hoàng|100%|lộ clip|thần kỳ|bài thuốc)', title, re.IGNORECASE))
            has_excl = "!" in title
            has_upper = any(w.isupper() and len(w) > 2 for w in title.split())
            sensational_cues = (0.3 if has_clickbait_words else 0.0) + (0.2 if has_excl else 0.0) + (0.2 if has_upper else 0.0)

        # Nhận diện các tuyên bố bất thường cực đoan hoặc lừa đảo phishing mã OTP / tặng tiền
        has_breaking_curfew = bool(re.search(r'(giới nghiêm|khẩn cấp|phong tỏa|tạm dừng toàn bộ giao thông)', title, re.IGNORECASE))
        has_phishing_scam = bool(re.search(r'(nhập.*otp|mã otp|bấm link|nhận.*tiền mặt|tri ân.*nhận ngay|\.online|\.xyz)', title + " " + content, re.IGNORECASE))

        # Đánh giá mức phạt:
        if source_count >= 0.2 and max_sim >= 0.45:
            # Có nguồn chính thống xác nhận sự kiện
            anomaly_penalty = float(np.clip(sensational_cues * 0.10 * (1.0 - max_sim), 0.0, 0.15))
        else:
            # Không có cơ quan báo chí chính thống nào xác nhận:
            if has_breaking_curfew:
                # Tuyên bố an ninh quốc gia (lệnh giới nghiêm) mà 0 báo nào xác nhận -> Phạt tối đa!
                anomaly_penalty = 0.95
            elif has_phishing_scam:
                # Dấu hiệu lừa đảo tài chính / OTP -> Phạt tối đa!
                anomaly_penalty = 0.90
            elif sensational_cues >= 0.20:
                # Giật gân câu view mà 0 có nguồn xác nhận -> Phạt nặng
                anomaly_penalty = float(np.clip(0.65 + sensational_cues * 0.35, 0.0, 1.0))
            else:
                # Tin tức trung tính bình thường mà chỉ thiếu đối chiếu -> Phạt nhẹ
                anomaly_penalty = float(np.clip((1.0 - max_sim) * 0.15, 0.0, 0.20))

        feature_vector = np.array([
            np.clip(max_sim, 0.0, 1.0),
            np.clip(source_count, 0.0, 1.0),
            np.clip(headline_stance, 0.0, 1.0),
            np.clip(anomaly_penalty, 0.0, 1.0)
        ], dtype=np.float32)

        return feature_vector, evidence_list


if __name__ == "__main__":
    verifier = CrossSourceVerifier(enable_web_search=True)
    t = "Chính phủ ban hành lệnh giới nghiêm và tạm dừng toàn bộ giao thông công cộng từ 22h đêm"
    c = "Theo văn bản khẩn, toàn bộ xe buýt sẽ dừng hoạt động."
    vec, ev = verifier.verify_article(t, c)
    print("Features for fake curfew:", vec)
    if ev:
        print("Top evidence:", ev[0]["source"], "-", ev[0]["title"], "sim:", ev[0]["similarity"])
