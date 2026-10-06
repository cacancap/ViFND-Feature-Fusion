# ViFND-Feature-Fusion: Phát Hiện Tin Giả Tiếng Việt Qua Hợp Nhất Đặc Trưng & XAI

Hệ thống phát hiện tin tức giả mạo, giật gân trên không gian mạng tiếng Việt bằng cách hợp nhất hai luồng biểu diễn:
1. **Biểu diễn Ngữ nghĩa Sâu (Deep Contextual Semantics)**: Mô hình ngôn ngữ `vinai/phobert-base-v2` (768 chiều).
2. **Biểu diễn Phong cách Hình thức (Surface Stylometrics)**: 15 chỉ số lượng hóa phong cách giật gân, câu view.
3. **Mô hình Hợp nhất Muộn (Late Fusion)**: Mạng nơ-ron đa đầu vào PyTorch (`DualStreamClassifier`) & Cây quyết định tăng cường (`XGBoost`).
4. **Trí tuệ Nhân tạo có thể Giải thích (Explainable AI)**: `SHAP TreeExplainer` minh giải nguyên nhân phân loại.
5. **Giao diện Trực quan**: Web UI Gradio dán URL bài báo, đo độ tin cậy và hiển thị biểu đồ SHAP Waterfall trong <1.5s.

---

## 1. TRIẾT LÝ VẬN HÀNH DỰ ÁN: CONTRACT-FIRST & ZERO-BLOCKING

Dự án áp dụng triệt để kiến trúc **Contract-First & Mock-Driven**:
- Mọi hợp đồng giao tiếp dữ liệu giữa 5 thành viên được đóng băng ngay tại `docs/SYSTEM_ARCHITECTURE_AND_ROLE_SPECIFICATIONS.md`.
- Thư mục `data/fixtures/` cung cấp sẵn dữ liệu mẫu chuẩn schema giúp **cả 5 thành viên có thể bắt tay vào lập trình độc lập 100% từ Ngày 1**, không ai phải chờ ai:
  - **TV1**: Xây dựng bộ cào và MinHash LSH đối chiếu với `data/fixtures/sample_corpus.parquet`.
  - **TV2**: Lập trình PyVi và 15 công thức Stylometrics sử dụng `data/fixtures/sample_corpus.parquet`.
  - **TV3**: Lập trình PhoBERT mean pooling trên Colab T4 sử dụng `data/fixtures/sample_corpus.parquet`.
  - **TV4**: Dựng mạng PyTorch và XGBoost sử dụng `data/fixtures/sample_X_stylometric.npy` và `sample_X_semantic.npy`.
  - **TV5**: Dựng giao diện Gradio và API FastAPI sử dụng `data/fixtures/sample_prediction_output.json`.

---

## 2. PHÂN CÔNG VAI TRÒ & TÀI LIỆU ĐẶC TẢ CHI TIẾT

| Thành Viên | Vai Trò & Trách Nhiệm | Thư Mục Chính | Tài Liệu Đặc Tả Chi Tiết |
|:---:|---|---|---|
| **TV1** | **Kỹ sư Dữ liệu & Cào Web (Data Eng)**: Cào tin đa nguồn, khử trùng lặp MinHash LSH, xuất `corpus_dedup.parquet`. | `src/crawlers/` | [`docs/ROLE_01_DATA_ENGINEERING_SPEC.md`](docs/ROLE_01_DATA_ENGINEERING_SPEC.md) |
| **TV2** | **Kỹ sư Ngôn ngữ & Stylometrics**: Tách từ ghép PyVi, lượng hóa 15 đặc trưng hình thức, xuất `X_stylometric.npy` $(N \times 15)$. | `src/features/` | [`docs/ROLE_02_STYLES_AND_LINGUISTICS_SPEC.md`](docs/ROLE_02_STYLES_AND_LINGUISTICS_SPEC.md) |
| **TV3** | **Kỹ sư Học sâu PhoBERT**: Cắt tỉa Head-Tail 256 tokens, GPU Mean Pooling, xuất `X_semantic.npy` $(N \times 768)$. | `src/features/` | [`docs/ROLE_03_PHOBERT_EMBEDDINGS_SPEC.md`](docs/ROLE_03_PHOBERT_EMBEDDINGS_SPEC.md) |
| **TV4** | **Trưởng nhóm Mô hình & XAI**: Mạng PyTorch đa đầu vào, XGBoost Late Fusion, Optuna tuning, SHAP TreeExplainer. | `src/models/` | [`docs/ROLE_04_MODELING_AND_XAI_SPEC.md`](docs/ROLE_04_MODELING_AND_XAI_SPEC.md) |
| **TV5** | **Kỹ sư MLOps & Web UI**: Giao diện Gradio 3 khối, API FastAPI, đóng gói Docker, điều phối slide bảo vệ 15 phút. | `src/ui/`, `src/api/` | [`docs/ROLE_05_MLOPS_AND_WEB_UI_SPEC.md`](docs/ROLE_05_MLOPS_AND_WEB_UI_SPEC.md) |

---

## 3. HƯỚNG DẪN KHỞI ĐỘNG NHANH CHO CÁC THÀNH VIÊN

### Cài đặt môi trường (Conda)
```bash
conda activate ML_Practice01
pip install -r requirements.txt
```

### Khởi động Giao diện Web Gradio (TV5)
```bash
python app.py
```
Truy cập tại: `http://localhost:7860` để trải nghiệm dán link bài báo, phân loại tin tức và xem biểu đồ SHAP Waterfall.

### Khởi động Dịch vụ REST API FastAPI (TV5)
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Truy cập Swagger UI tại: `http://localhost:8000/docs`

### Chạy Bộ Kiểm thử Toàn diện (Contract & Component Tests)
Bất kỳ thành viên nào trước khi commit code hoặc bàn giao dữ liệu chỉ cần chạy:
```bash
pytest tests/ -v
```
Nếu màn hình báo `9 passed` (100%) là toàn bộ module và cấu trúc dữ liệu đã chuẩn xác và sẵn sàng tích hợp!

### Tái tạo lại dữ liệu mẫu (Mock Fixtures)
```bash
python src/utils/mock_data_generator.py
```

### Triển khai với Docker & Docker Compose
```bash
docker-compose up --build
```

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```
ViFND-Feature-Fusion/
├── configs/guardrails.yaml          # Cấu hình tài nguyên và siêu tham số
├── data/
│   ├── raw/                         # Dữ liệu cào thô (TV1)
│   ├── processed/                   # Dữ liệu sạch & ma trận đặc trưng
│   └── fixtures/                    # Dữ liệu khuôn mẫu hợp đồng Ngày 1
├── docs/                            # Toàn bộ tài liệu đặc tả kiến trúc & 5 roles
├── src/
│   ├── crawlers/                    # Code cào & khử trùng lặp MinHash LSH (TV1)
│   ├── features/                    # Code PyVi, 15 Stylometrics (TV2) & PhoBERT (TV3)
│   ├── models/                      # Code PyTorch Multi-Input, XGBoost & SHAP (TV4)
│   ├── ui/                          # Code Gradio Web UI (TV5)
│   ├── api/                         # Code FastAPI Microservice (TV5)
│   └── utils/                       # Mock generator & utilities
├── tests/test_data_contracts.py     # Bộ kiểm thử hợp đồng tự động
├── requirements.txt                 # Thư viện phụ thuộc
└── README.md                        # Tài liệu này
```
