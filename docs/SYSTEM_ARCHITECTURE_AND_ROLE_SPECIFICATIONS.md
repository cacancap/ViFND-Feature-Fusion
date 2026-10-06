# TÀI LIỆU ĐẶC TẢ KIẾN TRÚC HỆ THỐNG VÀ QUY CHUẨN VAI TRÒ NHÓM 5 NGƯỜI
## DỰ ÁN: ViFND-Feature-Fusion (Phát Hiện Tin Giả Tiếng Việt Qua Hợp Nhất Đặc Trưng & XAI)

**Cập nhật lần cuối**: Tháng 10/2026  
**Mục tiêu dự án**: Xây dựng hệ thống học máy phát hiện tin giả, tin giật gân, câu view trên báo mạng và mạng xã hội tiếng Việt bằng cách kết hợp biểu diễn ngữ nghĩa sâu (PhoBERT 768 chiều) với 15 chỉ số phong cách hình thức (Stylometrics), tích hợp giải thích quyết định bằng SHAP và triển khai Web UI Gradio.

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG 6 TẦNG (6-TIER ARCHITECTURE)

```
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: THU THẬP ĐA NGUỒN & TRUY XUẤT XUẤT XỨ (TV1)                   │
│ - RSS Báo chính thống: VnExpress, Tuổi Trẻ, VietnamNet (Nhãn 0)        │
│ - Blog / Diễn đàn giật gân & Bộ benchmark VFND (Nhãn 1)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: LÀM SẠCH & KHỬ TRÙNG LẶP CẬN ÂM MINHASH LSH (TV1)              │
│ - Bóc tách HTML boilerplate, quảng cáo; Chuẩn hóa Unicode NFC          │
│ - MinHash (128 hàm băm), LSH Index (b=32, r=4, Ngưỡng Jaccard >= 0.85) │
│ - Xuất file: data/processed/corpus_dedup.parquet                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: TÁCH TỪ GHÉP TIẾNG VIỆT CHUYÊN SÂU (TV2)                       │
│ - PyVi ViTokenizer: Tách từ đơn thành từ ghép ('học_sinh', 'chấn_động')│
│ - Xuất file: data/processed/corpus_tokenized.parquet                   │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│ TẦNG 4A: BIỂU DIỄN NGỮ NGHĨA SÂU     │  │ TẦNG 4B: TRÍCH XUẤT HÌNH THỨC│
│ (TV3 - PhoBERT Specialist)           │  │ (TV2 - Stylometrics Engineer)│
│ - Mô hình: vinai/phobert-base-v2     │  │ - 15 công thức toán học      │
│ - Cắt tỉa Head-Tail ([1:128]+[L-128:L)│  │ - Mật độ (!, ?, chữ hoa, từ  │
│ - Attention-Masked Mean Pooling      │  │   khóa giật gân, tỷ lệ đề/bài│
│ - Chuẩn hóa vector đơn vị L2         │  │ - Chuẩn hóa StandardScaler   │
│ - Xuất: X_semantic.npy (N x 768)     │  │ - Xuất: X_stylometric.npy    │
│                                      │  │         (N x 15)             │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    └────────────────┬───────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5: HỢP NHẤT ĐẶC TRƯNG MUỘN (LATE FUSION) & MÔ HÌNH HÓA (TV4)      │
│ - Mô hình A: Ghép vector 783 chiều + XGBoost / CatBoost                │
│ - Mô hình B: Mạng nơ-ron đa đầu vào PyTorch (DualStreamClassifier)     │
│ - Đánh giá: Stratified 5-Fold CV, Tối ưu siêu tham số bằng Optuna      │
│ - Xuất checkpoint: best_fusion_xgboost.json, best_fusion_pytorch.pt    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 6: TRÍ TUỆ NHÂN TẠO GIẢI THÍCH ĐƯỢC (XAI) & DEMO WEB (TV4, TV5)  │
│ - TV4: SHAP TreeExplainer bóc tách % rủi ro từng đặc trưng             │
│ - TV5: Web Demo Gradio (Dán URL, đo rủi ro, vẽ biểu đồ SHAP Waterfall) │
│ - TV5: Dịch vụ REST API FastAPI (/predict, /explain) + Dockerfile      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TRIẾT LÝ VẬN HÀNH "CONTRACT-FIRST & MOCK-DRIVEN" (ZERO-BLOCKING)

### 2.1 Tại Sao Nhóm Cần Triết Lý Này?
Mô hình làm việc truyền thống kiểu thác nước (Waterfall) sẽ khiến:
- TV2 & TV3 phải chờ TV1 cào dữ liệu xong (mất 2-4 tuần).
- TV4 phải chờ TV2 & TV3 trích xuất đặc trưng xong (mất thêm 2 tuần).
- TV5 chỉ bắt đầu làm web vào 1-2 tuần cuối cùng trước ngày nộp báo cáo.

### 2.2 Giải Pháp Của Chúng Ta:
1. **Đóng băng Hợp đồng (Freeze Data Contracts)** ngay ngày đầu tiên.
2. Cung cấp bộ dữ liệu giả lập chuẩn schema trong thư mục `data/fixtures/`:
   - `sample_corpus.parquet`: Giúp TV2 & TV3 viết code tiền xử lý ngay.
   - `sample_X_stylometric.npy` & `sample_X_semantic.npy`: Giúp TV4 dựng mạng PyTorch và XGBoost ngay.
   - `sample_prediction_output.json`: Giúp TV5 dựng Web UI Gradio và API FastAPI ngay.
3. **Cả 5 thành viên bắt đầu lập trình độc lập 100% từ Ngày 1**.

---

## 3. BẢNG TỔNG HỢP HỢP ĐỒNG GIAO TIẾP DỮ LIỆU (DATA CONTRACTS MATRIX)

| Mã Hợp Đồng | Bên Giao (Producer) | Bên Nhận (Consumer) | File Bàn Giao Thật | File Mẫu Ngày 1 (Fixture) | Ràng Buộc Kỹ Thuật Bắt Buộc |
|---|:---:|:---:|---|---|---|
| **Contract 1** | **TV1** | **TV2, TV3** | `data/processed/corpus_dedup.parquet` | `data/fixtures/sample_corpus.parquet` | Bảng 6 cột chuẩn (`doc_id`, `title`, `content`, `source`, `url`, `label`). Không chứa giá trị null. |
| **Contract 2** | **TV2** | **TV4** | `data/processed/features/X_stylometric.npy` | `data/fixtures/sample_X_stylometric.npy` | Tensor 2D kích thước $(N, 15)$, kiểu `float32`, chuẩn hóa `StandardScaler` ($\mu=0, \sigma=1$). Không chứa NaN/Inf. |
| **Contract 3** | **TV3** | **TV4** | `data/processed/features/X_semantic.npy` | `data/fixtures/sample_X_semantic.npy` | Tensor 2D kích thước $(N, 768)$, kiểu `float32`, chuẩn hóa vector đơn vị L2 ($\|\mathbf{z}\|_2 = 1.0$). |
| **Contract 4** | **TV4** | **TV5** | `src/models/fusion_classifier.py` | `data/fixtures/sample_prediction_output.json` | Hàm `predict_with_shap(title, content)` trả về Dict đúng cấu trúc: `prediction`, `probability_fake`, `top_shap_factors`. |

---

## 4. QUY TRÌNH KIỂM THỬ TỰ ĐỘNG CHẤT LƯỢNG (AUTOMATED QA)

Trước khi đẩy code lên nhánh `develop` hoặc bàn giao dữ liệu cho thành viên khác, mỗi bạn chỉ cần chạy 1 lệnh duy nhất:
```bash
pytest tests/test_data_contracts.py
```
Bộ kiểm thử sẽ tự động kiểm tra:
1. File parquet có đủ 6 cột và đúng kiểu dữ liệu hay không.
2. Ma trận stylometric có đúng 15 cột, đã chuẩn hóa và không có giá trị bất thường hay không.
3. Ma trận semantic có đúng 768 cột và đạt chuẩn vector đơn vị L2 hay không.
4. Đầu ra của hàm dự đoán có tương thích với giao diện Web hay không.

---

## 5. CẤU TRÚC THƯ MỤC CHUẨN CỦA REPO `ViFND-Feature-Fusion`

```
ViFND-Feature-Fusion/
├── configs/
│   └── guardrails.yaml              # Cấu hình tài nguyên & siêu tham số
├── data/
│   ├── raw/                         # Dữ liệu cào thô (TV1)
│   ├── processed/                   # Dữ liệu sạch, tách từ, ma trận đặc trưng
│   │   └── features/                # Nơi lưu X_stylometric.npy và X_semantic.npy
│   └── fixtures/                    # DỮ LIỆU MẪU ĐÓNG BĂNG HỢP ĐỒNG NGÀY 1
├── docs/
│   ├── SYSTEM_ARCHITECTURE_AND_ROLE_SPECIFICATIONS.md # Tài liệu tổng quan này
│   ├── ROLE_01_DATA_ENGINEERING_SPEC.md              # Đặc tả chi tiết TV1
│   ├── ROLE_02_STYLES_AND_LINGUISTICS_SPEC.md        # Đặc tả chi tiết TV2
│   ├── ROLE_03_PHOBERT_EMBEDDINGS_SPEC.md            # Đặc tả chi tiết TV3
│   ├── ROLE_04_MODELING_AND_XAI_SPEC.md              # Đặc tả chi tiết TV4
│   └── ROLE_05_MLOPS_AND_WEB_UI_SPEC.md              # Đặc tả chi tiết TV5
├── src/
│   ├── crawlers/                    # Code cào & MinHash LSH (TV1)
│   ├── features/                    # Code Stylometrics (TV2) & PhoBERT (TV3)
│   ├── models/                      # Code Late Fusion & SHAP (TV4)
│   ├── ui/                          # Code Gradio & FastAPI (TV5)
│   └── utils/                       # Reusable utils, mock generator, loggers
├── tests/
│   └── test_data_contracts.py       # Bộ kiểm thử hợp đồng tự động
├── outputs/
│   ├── checkpoints/                 # Trọng số mô hình đã huấn luyện
│   ├── logs/                        # Nhật ký thực thi
│   └── reports/                     # Báo cáo đánh giá JSON/Markdown
├── app.py                           # Điểm chạy Web Gradio chính
├── requirements.txt                 # Danh sách thư viện Python
└── README.md                        # Hướng dẫn khởi động nhanh
```
