# ĐẶC TẢ VAI TRÒ & QUY TRÌNH KỸ THUẬT: THÀNH VIÊN 4 (TV4)
## TRƯỞNG NHÓM MÔ HÌNH HÓA, TỐI ƯU & XAI (Late Fusion Modeling, Optimization & XAI Lead)

**Phạm vi phụ trách**: Tầng 5 (Hợp nhất đặc trưng muộn - Late Fusion) & Tầng 6 (Minh giải mô hình XAI bằng SHAP).  
**Thư mục làm việc chính**: `src/models/`, `outputs/checkpoints/`, `outputs/reports/`  
**Tập tin bàn giao chính**: 
1. Checkpoints mô hình: `outputs/checkpoints/best_fusion_xgboost.json` & `best_fusion_pytorch.pt`  
2. Class dự đoán & giải thích: `src/models/fusion_classifier.py`  
3. Báo cáo đánh giá hiệu năng: `outputs/reports/model_benchmark_results.json`  

---

## 1. MỤC TIÊU VÀ SỨ MỆNH KHOA HỌC
Sứ mệnh của TV4 là chứng minh luận điểm cốt lõi của toàn bộ đề tài:
1. **Hợp nhất đặc trưng (Feature Fusion)** giúp cải thiện vượt bậc hiệu năng so với các mô hình đơn luồng (TF-IDF đơn lẻ, PhoBERT đơn lẻ, Stylometric đơn lẻ).
2. Xóa bỏ định kiến "Hộp đen" (Black-Box AI) bằng cách sử dụng **SHAP (SHapley Additive exPlanations)** để chứng minh toán học nguyên nhân mô hình ra quyết định.

---

## 2. QUY TRÌNH KỸ THUẬT CỦA TV4

### Bước 1: Tiếp Nhận Dữ Liệu & Thiết Lập Baseline
- Tiếp nhận ma trận $X_{\text{sty}} \in \mathbb{R}^{N \times 15}$ từ TV2 và $X_{\text{sem}} \in \mathbb{R}^{N \times 768}$ từ TV3 (sử dụng các file mẫu trong `data/fixtures/` khi phát triển ban đầu).
- Huấn luyện 3 mô hình đối chứng (Baselines) qua 5-Fold Cross Validation:
  - Baseline 1: TF-IDF (1-2 gram, max_features=500) + Logistic Regression.
  - Baseline 2: Chỉ dùng 15 đặc trưng Stylometrics + XGBoost.
  - Baseline 3: Chỉ dùng 768 chiều PhoBERT + XGBoost.

### Bước 2: Xây Dựng 2 Kiến Trúc Hợp Nhất Đặc Trưng Muộn (Late Fusion)

#### Kiến trúc A: Cây quyết định tăng cường trên không gian ghép nối (Concatenated XGBoost / CatBoost)
- Ghép nối trực tiếp hai ma trận thành vector 783 chiều:
  $$\mathbf{X}_{\text{fused}} = \left[ \mathbf{X}_{\text{sem}} \;\|\; \mathbf{X}_{\text{sty}} \right] \in \mathbb{R}^{N \times 783}$$
- Cấu hình XGBoost xử lý mất cân bằng lớp bằng tham số `scale_pos_weight = \frac{N_{\text{neg}}}{N_{\text{pos}}}`.

#### Kiến trúc B: Mạng nơ-ron sâu đa đầu vào PyTorch (`DualStreamClassifier`)
Mạng gồm 2 nhánh song song giải quyết triệt để sự mất cân xứng chiều (768 chiều vs 15 chiều):
```python
class DualStreamClassifier(nn.Module):
    def __init__(self, sem_dim=768, sty_dim=15):
        super().__init__()
        # Nhánh 1: Nén đặc trưng ngữ nghĩa
        self.sem_branch = nn.Sequential(
            nn.Linear(sem_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Nhánh 2: Mở rộng đặc trưng hình thức phong cách
        self.sty_branch = nn.Sequential(
            nn.Linear(sty_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Tầng Hợp nhất & Phân loại
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### Bước 3: Tối Ưu Hóa Siêu Tham Số Bằng Optuna (Hyperparameter Tuning)
- Thiết lập quy trình đánh giá chuẩn: **Stratified 5-Fold Cross Validation** (đảm bảo tỷ lệ nhãn ở mỗi fold là như nhau).
- Chạy 50 trials với thư viện Optuna để tối ưu cho XGBoost:
  - `learning_rate` $\in [0.01, 0.2]$
  - `max_depth` $\in [3, 8]$
  - `subsample` $\in [0.6, 1.0]$
  - `colsample_bytree` $\in [0.6, 1.0]$
- Mục tiêu tối ưu: Tối đa hóa **Macro-F1 Score** trên tập validation.

### Bước 4: Trí Tuệ Nhân Tạo Có Thể Giải Thích (Explainable AI - SHAP)
- Sử dụng `shap.TreeExplainer` trên mô hình XGBoost tốt nhất:
  ```python
  import shap
  explainer = shap.TreeExplainer(best_xgb_model)
  shap_values = explainer.shap_values(X_test_fused)
  ```
- **Xuất 2 sản phẩm XAI cốt lõi**:
  1. *Biểu đồ tầm quan trọng toàn cục (Summary Plot)*: Chứng minh 15 đặc trưng Stylometrics đóng góp bao nhiêu % tầm ảnh hưởng so với các đặc trưng từ vựng/ngữ nghĩa.
  2. *Hàm giải thích cục bộ cho từng bài viết (Local Waterfall Plot)*: Trả về danh sách các đặc trưng kéo rủi ro bài viết về phía "Fake" hoặc "Real".

### Bước 5: Đóng Gói Hàm Dự Đoán Dành Cho TV5
TV4 chịu trách nhiệm xây dựng class `FakeNewsPredictor` trong `src/models/fusion_classifier.py` xuất ra hàm:
```python
def predict_with_shap(title: str, content: str) -> dict:
    """
    Nhận tiêu đề và nội dung bài báo, thực thi toàn bộ pipeline và trả về:
    {
        "status": "success",
        "prediction": "TIN GIẢ / GIẬT GÂN", # hoặc "TIN THẬT / XÁC THỰC"
        "probability_fake": 0.9412,
        "confidence_score": 94.1,
        "risk_tier": "CRITICAL",            # "LOW", "MODERATE", "CRITICAL"
        "top_shap_factors": [
            {"feature": "exclamation_density", "contribution": +0.48, "description_vi": "Mật độ dấu chấm than (!) cao bất thường"},
            {"feature": "clickbait_keyword_hits", "contribution": +0.35, "description_vi": "Chứa từ khóa giật gân kích động"},
            ...
        ]
    }
    """
```

---

## 3. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE - DOD)
- [ ] Vượt qua test tự động: `pytest tests/test_data_contracts.py::test_contract_p4`.
- [ ] Mô hình Late Fusion đạt **Macro-F1 $\ge 0.90$** và **PR-AUC $\ge 0.92$** trên tập kiểm thử phân tầng độc lập.
- [ ] Chứng minh được Late Fusion vượt trội hơn mô hình Baseline TF-IDF tối thiểu **+10% Macro-F1**.
- [ ] Checkpoints được lưu đầy đủ tại `outputs/checkpoints/`.
- [ ] Hàm `predict_with_shap` thực thi nhanh trong thời gian $< 50\text{ms}$ đối với một mẫu đầu vào.
