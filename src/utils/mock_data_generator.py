"""
Mock Data Fixtures Generator for ViFND-Feature-Fusion
Generates synthetic data fixtures adhering to strict team data contracts:
- Contract 1: sample_corpus.parquet (Schema: doc_id, title, content, source, url, label)
- Contract 2: sample_X_stylometric.npy (Shape: [N, 15], StandardScaler normalized)
- Contract 3: sample_X_semantic.npy (Shape: [N, 768], L2 normalized)
- Contract 4: sample_prediction_output.json (Schema: prediction, probability_fake, risk_tier, top_shap_factors)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

FEATURE_NAMES_STYLO = [
    "upper_ratio",
    "exclamation_density",
    "question_density",
    "punct_density",
    "title_body_ratio",
    "clickbait_keyword_hits",
    "digit_in_title",
    "colon_dash_density",
    "quote_density",
    "avg_sentence_len",
    "urgency_density",
    "pronoun_density",
    "teencode_ratio",
    "title_body_jaccard",
    "sentiment_extremity"
]

def generate_fixtures(output_dir: str = "data/fixtures", num_samples: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Contract 1: Corpus (Parquet + CSV)
    real_titles = [
        "Chính phủ ban hành nghị định về quản lý và phát triển hạ tầng giao thông đô thị",
        "Bộ Y tế triển khai chiến dịch tiêm chủng mở rộng phòng chống sốt xuất huyết 2026",
        "Ngân hàng Nhà nước giữ vững ổn định tỷ giá và thanh khoản hệ thống tín dụng",
        "Việt Nam đẩy mạnh hợp tác chuyển đổi số và công nghệ bán dẫn với đối tác ASEAN",
        "Bộ Giáo dục và Đào tạo công bố quy chế thi tốt nghiệp trung học phổ thông mới"
    ]
    fake_titles = [
        "CHẤN ĐỘNG: Phát hiện phương thuốc nam chữa khỏi 100% ung thư trong 7 ngày!!!",
        "BÍ MẬT BỊ GIẤU KÍN: Uống nước lá này mỗi sáng để tiêu diệt mọi khối u ác tính???",
        "CẢNH BÁO NGUY HIỂM: Ăn mì tôm cùng trứng gà tạo chất kịch độc gây suy thận cấp!",
        "KIẾM 50 TRIỆU/NGÀY: Cơ hội đầu tư app tài chính quốc tế cam kết lợi nhuận khủng!",
        "LỘ CLIP NÓNG 15 PHÚT: Cả showbiz rung chuyển trước danh tính nữ ca sĩ hạng A!"
    ]
    
    records = []
    for i in range(num_samples):
        is_fake = 1 if i % 2 == 1 else 0
        title = fake_titles[(i // 2) % len(fake_titles)] if is_fake else real_titles[(i // 2) % len(real_titles)]
        title = f"{title} (Mã tin #{i+1000})"
        content = (
            f"Nội dung chi tiết của bản tin số #{i+1000}. "
            + ("Bà con hãy chia sẻ ngay bài viết này trước khi bị gỡ bỏ hoàn toàn khỏi mạng xã hội! Bác sĩ bệnh viện lớn không muốn bạn biết điều này. " if is_fake else "Theo thông cáo báo chí chính thức từ cơ quan chức năng, tiến độ thực hiện đang diễn ra đúng kế hoạch và đạt các chỉ tiêu đề ra. ")
            + "Các chuyên gia khuyến cáo độc giả nên theo dõi các kênh thông tin chính thống để nắm bắt diễn biến chính xác nhất."
        )
        records.append({
            "doc_id": f"VNNEWS_{i:06d}",
            "title": title,
            "content": content,
            "source": "Unverified_Social_Blog" if is_fake else "VnExpress_ThoiSu",
            "url": f"https://mock-source.vn/news/{i:06d}",
            "crawled_at": "2026-10-06T10:00:00Z",
            "label": np.int8(is_fake)
        })
        
    df = pd.DataFrame(records)
    parquet_path = os.path.join(output_dir, "sample_corpus.parquet")
    csv_path = os.path.join(output_dir, "sample_corpus.csv")
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[Fixture 1] Sample Corpus created: {parquet_path} ({len(df)} rows)")

    # 2. Contract 2: Stylometrics (N, 15)
    X_sty = np.random.normal(0.0, 1.0, size=(num_samples, 15)).astype(np.float32)
    # Inject signal for fake news (odd indices)
    X_sty[1::2, 0] += 1.8 # higher uppercase ratio
    X_sty[1::2, 1] += 2.5 # higher exclamation density
    X_sty[1::2, 5] += 3.0 # higher clickbait keywords
    
    sty_path = os.path.join(output_dir, "sample_X_stylometric.npy")
    names_path = os.path.join(output_dir, "stylometric_feature_names.json")
    np.save(sty_path, X_sty)
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES_STYLO, f, indent=2, ensure_ascii=False)
    print(f"[Fixture 2] Sample Stylometrics created: {sty_path} (Shape: {X_sty.shape})")

    # 3. Contract 3: Semantic Embeddings (N, 768)
    X_sem = np.random.normal(0.0, 1.0, size=(num_samples, 768)).astype(np.float32)
    norms = np.linalg.norm(X_sem, axis=1, keepdims=True)
    X_sem = X_sem / (norms + 1e-8) # L2 normalization
    
    sem_path = os.path.join(output_dir, "sample_X_semantic.npy")
    np.save(sem_path, X_sem)
    print(f"[Fixture 3] Sample Semantic Embeddings created: {sem_path} (Shape: {X_sem.shape})")

    # 4. Contract 4: Prediction Output JSON
    sample_prediction = {
        "status": "success",
        "url": "https://tin-tuc-mang-xa-hoi.vn/post/10293",
        "title": "CHẤN ĐỘNG: Phát hiện bài thuốc chữa khỏi 100% ung thư trong 7 ngày!!!",
        "prediction": "TIN GIẢ / GIẬT GÂN",
        "probability_fake": 0.9412,
        "confidence_score": 94.1,
        "risk_tier": "CRITICAL",
        "stylometric_summary": {
            "exclamation_density": 0.045,
            "clickbait_keyword_hits": 4,
            "upper_ratio": 0.182,
            "title_body_ratio": 0.150
        },
        "top_shap_factors": [
            {
                "feature": "exclamation_density",
                "contribution": 0.482,
                "direction": "INCREASES_FAKE_RISK",
                "description_vi": "Mật độ dấu chấm than (!) cao bất thường (3 dấu chấm than liên tiếp)."
            },
            {
                "feature": "clickbait_keyword_hits",
                "contribution": 0.351,
                "direction": "INCREASES_FAKE_RISK",
                "description_vi": "Chứa các từ khóa giật gân, kích động cảm xúc ('CHẤN ĐỘNG', '100%')."
            },
            {
                "feature": "upper_ratio",
                "contribution": 0.124,
                "direction": "INCREASES_FAKE_RISK",
                "description_vi": "Tỷ lệ ký tự viết hoa trong tiêu đề cao bất thường."
            },
            {
                "feature": "phobert_semantics",
                "contribution": -0.016,
                "direction": "SLIGHTLY_DECREASES_RISK",
                "description_vi": "Không gian ngữ nghĩa bài viết có cấu trúc tương đối liền mạch."
            }
        ]
    }
    pred_path = os.path.join(output_dir, "sample_prediction_output.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(sample_prediction, f, indent=2, ensure_ascii=False)
    print(f"[Fixture 4] Sample Prediction Output created: {pred_path}")
    print("\n[SUCCESS] All team fixtures generated successfully!")

if __name__ == "__main__":
    generate_fixtures()
