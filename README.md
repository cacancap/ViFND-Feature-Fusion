# ViFND-Feature-Fusion: Hệ Thống Phát Hiện Tin Giả Tiếng Việt Qua Hợp Nhất Đa Luồng (Tri-Stream Late Fusion) & XAI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 11 Passed](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()
[![Model: Tri--Stream Fusion](https://img.shields.io/badge/Model-Tri--Stream%20787D-orange.svg)]()

Hệ thống phát hiện và cảnh báo thông tin sai lệch, tin tức giả mạo và giật gân trên không gian mạng tiếng Việt. Dự án tiếp cận theo hướng **Hợp nhất Muộn Ba Luồng (Tri-Stream Late Fusion - 787 chiều)** kết hợp **Trí tuệ nhân tạo có thể giải thích (Explainable AI - SHAP)**:

1. **Luồng 1: Biểu diễn Ngữ nghĩa Sâu (Deep Contextual Semantics - 768 chiều)**: Sử dụng mô hình ngôn ngữ tiếng Việt tiền huấn luyện `vinai/phobert-base-v2` kết hợp chiến lược Head-Tail Truncation và Mean Pooling (chuẩn hóa $L_2$).
2. **Luồng 2: Biểu diễn Phong cách Hình thức (Surface Stylometrics - 15 chiều)**: Lượng hóa 15 đặc trưng phong cách viết (mật độ từ ngữ giật gân, câu view, dấu câu bất thường, tỷ lệ viết hoa...) đã qua chuẩn hóa Z-score (`StandardScaler`).
3. **Luồng 3: Tầng Kiểm chứng Chéo Đa Nguồn (Cross-Source Corroboration - 4 chiều)**: Đối soát tức thời sự kiện với báo chí chính thống tiếng Việt (VnExpress, VTV, Dân Trí, Tuổi Trẻ, VietnamNet...) qua cơ chế Hybrid (Local Parquet Index + Google News RSS), tính toán độ tương đồng nội dung, lập trường tiêu đề và áp điểm phạt bất thường đối với thông tin nhạy cảm không được kiểm chứng.
4. **Mô hình Hợp nhất Muộn (Late Fusion)**: Kết hợp Cây quyết định tăng cường (`XGBoost`) có cơ chế kiểm soát chống quá khớp (subsampling `colsample_bynode=0.40`) và Mạng nơ-ron đa đầu vào PyTorch (`DualStreamClassifier`).
5. **Minh giải Khoa học (XAI)**: Sử dụng `SHAP TreeExplainer` bóc tách mức độ tác động của từng đặc trưng lên quyết định phân loại, hiển thị cùng danh sách bằng chứng đối chiếu xác thực.

---

## 1. KHUNG LÝ LUẬN KHOA HỌC: ĐỊNH NGHĨA "TIN GIẢ"

Nhằm đảm bảo tính chính xác về mặt học thuật và pháp lý, dự án áp dụng **Khung lý luận Rối loạn Thông tin (Information Disorder Framework)** do Hội đồng châu Âu (Council of Europe) và Harvard Shorenstein Center công bố, phân định rõ:

- **Bịa đặt có chủ đích (Disinformation)**: Thông tin hoàn toàn sai sự thật, được ngụy tạo với động cơ trục lợi, phá hoại hoặc lừa đảo (ví dụ: bịa đặt văn bản giới nghiêm, dàn dựng tin tức giả mạo ngân hàng, tin y tế phản khoa học).
- **Thông tin sai lệch vô ý (Misinformation)**: Thông tin không chuẩn xác do hiểu lầm, truyền tai hoặc cắt ghép sai ngữ cảnh, dù không nhất thiết có ác ý ban đầu.
- **Tiêu đề câu view & Giật gân (Clickbait & Sensationalism)**: Nội dung có thể có một phần sự thật nhưng bị thổi phồng quá mức qua các thủ pháp giật gân, dùng dấu câu la hét (`!!!`), phóng đại tỷ lệ (`100%`) hoặc tiêu đề "treo đầu dê bán thịt chó".

Hệ thống ViFND không chỉ phân loại nhị phân (Thật/Giả) mà cung cấp **Thang đo rủi ro đa cấp (Low / Moderate / Critical)** kèm **Bằng chứng đối soát báo chí** nhằm hỗ trợ người dùng tự đưa ra quyết định độc lập.

---

## 2. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG

Hệ thống hỗ trợ đầy đủ 2 phương thức quản lý môi trường ảo phổ biến: **Conda** (khuyến nghị cho Khoa học Dữ liệu / Học máy) và **venv** (môi trường Python tiêu chuẩn). Bạn chỉ cần chọn **một trong hai cách** dưới đây:

### Cách 1: Sử dụng Anaconda / Miniconda (Khuyến nghị)

```bash
# 1. Tạo môi trường ảo Python 3.11 mới
conda create -n vifnd python=3.11 -y

# 2. Kích hoạt môi trường
conda activate vifnd

# 3. Nâng cấp pip và cài đặt toàn bộ thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý**: Nếu bạn đã có sẵn môi trường Conda cho môn học (ví dụ: `ML_Practice01`), chỉ cần kích hoạt và cài đặt:
> ```bash
> conda activate ML_Practice01
> pip install -r requirements.txt
> ```

---

### Cách 2: Sử dụng Python Virtual Environment (`venv`) Tiêu Chuẩn

Yêu cầu máy tính đã cài đặt Python $\ge 3.10$ (khuyến nghị Python 3.11).

#### Trên Windows (PowerShell / Command Prompt):
```powershell
# 1. Di chuyển vào thư mục gốc của dự án
cd D:\SourceCodes\Python_SourceCodes\ViFND-Feature-Fusion

# 2. Khởi tạo môi trường ảo .venv
python -m venv .venv

# 3. Kích hoạt môi trường ảo:
# Với PowerShell:
.\.venv\Scripts\Activate.ps1
# (Nếu gặp lỗi chính sách thực thi script trên Windows PowerShell, chạy trước: Set-ExecutionPolicy Unrestricted -Scope Process)

# Với Command Prompt (cmd):
.\.venv\Scripts\activate.bat

# 4. Nâng cấp pip và cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Trên Linux / macOS:
```bash
# 1. Khởi tạo môi trường ảo
python3 -m venv .venv

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate

# 3. Nâng cấp pip và cài đặt thư viện
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. QUY TRÌNH VẬN HÀNH & KIỂM THỬ THỰC NGHIỆM

Sau khi kích hoạt môi trường ảo, các thành viên có thể thực thi các tác vụ theo thứ tự sau:

### Bước 1: Huấn luyện mô hình hợp nhất & Xuất Checkpoint
Chạy quy trình huấn luyện 5-Fold Stratified Cross-Validation, tự động trích xuất đặc trưng đa luồng, chuẩn hóa Z-score và tối ưu hóa bộ phân loại:
```bash
python src/models/train_fusion.py
```
*Kết quả xuất ra*: Checkpoint mô hình [`best_fusion_xgboost.json`](outputs/checkpoints/best_fusion_xgboost.json), trọng số chuẩn hóa [`fusion_scaler.json`](outputs/checkpoints/fusion_scaler.json) và báo cáo thực nghiệm [`evaluation_metrics.json`](outputs/reports/evaluation_metrics.json).

### Bước 2: Chạy kiểm thử tự động toàn diện (Automated QA)
Đảm bảo tất cả hợp đồng dữ liệu, module trích xuất và mạng nơ-ron hoạt động chính xác 100%:
```bash
pytest tests/ -v
```
*(Yêu cầu chuẩn: `11 passed` trong thời gian <15 giây).*

### Bước 3: Đánh giá thử nghiệm trên 5 mẫu bài báo ngoài đời thực
Chạy kịch bản kiểm tra độc lập trên 5 bài báo đời thực (bài báo chính sách giáo dục, khoa học công nghệ, lừa đảo y tế, tin giả tinh vi không dùng dấu than, và lừa đảo tài chính):
```bash
python sandbox/test_random_articles.py
```

### Bước 4: Khởi chạy Giao diện Web Trực quan (Gradio Web UI)
Khởi động giao diện web tương tác hỗ trợ người dùng dán URL bài báo tự động cào, nhập văn bản tùy biến, hiển thị thang đo rủi ro và biểu đồ SHAP Waterfall:
```bash
python app.py
```
- Mở trình duyệt web và truy cập địa chỉ: **`http://127.0.0.1:7860`**
- Giao diện được tối ưu hóa hiển thị sắc nét (High Contrast) trên cả **Light Mode** và **Dark Mode**.

### Bước 5: Khởi chạy Microservice REST API (FastAPI)
Dành cho việc tích hợp vào hệ sinh thái ứng dụng khác (Extension trình duyệt, Bot mạng xã hội):
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```
- Tài liệu tương tác API Swagger UI: **`http://127.0.0.1:8000/docs`**
- Hỗ trợ endpoint: `POST /api/v1/predict` và `POST /api/v1/explain`.

---

## 4. BẢNG SO SÁNH HIỆU NĂNG THỰC NGHIỆM

Kết quả thực nghiệm trên tập dữ liệu chuẩn hóa thông qua **Stratified 5-Fold Cross-Validation**:

| Không Gian Đặc Trưng / Kiến Trúc | Số Chiều | Accuracy | Macro-F1 | ROC-AUC | Ghi Chú |
|---|:---:|:---:|:---:|:---:|---|
| **1. Baseline (Semantics 300D + Logistic Regression)** | 300 | 96.33% | 96.33% | 0.9957 | Mô hình cơ sở |
| **2. Pure Stylometrics (15D + XGBoost)** | 15 | 98.33% | 98.33% | 0.9961 | Chỉ dùng văn phong |
| **3. Pure Semantics (PhoBERT 768D + XGBoost)** | 768 | 96.50% | 96.49% | 0.9888 | Chỉ dùng ngữ nghĩa sâu |
| **4. Dual-Stream Fusion (PhoBERT + Stylo 783D)** | 783 | 98.67% | 98.67% | 0.9988 | Hợp nhất 2 luồng |
| **5. Tri-Stream Fusion (PhoBERT + Stylo + Verifier 787D)** | **787** | **100.00%** | **100.00%** | **1.0000** | **Đề xuất của dự án** |

---

## 5. PHÂN CÔNG VAI TRÒ & TÀI LIỆU ĐẶC TẢ CHI TIẾT

Dự án áp dụng mô hình phân tách vai trò độc lập theo triết lý **Contract-First**:

| Vai Trò | Trách Nhiệm Kỹ Thuật Chính | Thư Mục Phụ Trách | Tài Liệu Chi Tiết |
|:---:|---|---|---|
| **TV1** | **Kỹ sư Dữ liệu & Cào Web**: Thu thập dữ liệu đa nguồn, khử trùng lặp MinHash LSH, chuẩn hóa kho dữ liệu `corpus_dedup.parquet`. | `src/crawlers/` | [`docs/ROLE_01_DATA_ENGINEERING_SPEC.md`](docs/ROLE_01_DATA_ENGINEERING_SPEC.md) |
| **TV2** | **Kỹ sư Ngôn ngữ & Stylometrics**: Tách từ ghép PyVi, lượng hóa 15 chỉ số văn phong, tích hợp Z-score scaling. | `src/features/` | [`docs/ROLE_02_STYLES_AND_LINGUISTICS_SPEC.md`](docs/ROLE_02_STYLES_AND_LINGUISTICS_SPEC.md) |
| **TV3** | **Kỹ sư Học sâu PhoBERT**: Cắt tỉa Head-Tail 256 tokens, trích xuất vector ngữ nghĩa 768 chiều chuẩn $L_2$. | `src/features/` | [`docs/ROLE_03_PHOBERT_EMBEDDINGS_SPEC.md`](docs/ROLE_03_PHOBERT_EMBEDDINGS_SPEC.md) |
| **TV4** | **Trưởng nhóm Mô hình & XAI**: Xây dựng Tầng kiểm chứng đa nguồn (`cross_verifier.py`), mạng PyTorch, XGBoost Late Fusion và SHAP XAI. | `src/models/`, `src/features/` | [`docs/ROLE_04_MODELING_AND_XAI_SPEC.md`](docs/ROLE_04_MODELING_AND_XAI_SPEC.md) |
| **TV5** | **Kỹ sư MLOps & Web UI**: Xây dựng Giao diện Gradio 3 khối, Microservice FastAPI, đóng gói Docker và tài liệu kiểm thử. | `src/ui/`, `src/api/`, `app.py` | [`docs/ROLE_05_MLOPS_AND_WEB_UI_SPEC.md`](docs/ROLE_05_MLOPS_AND_WEB_UI_SPEC.md) |

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ViFND-Feature-Fusion/
├── configs/
│   └── guardrails.yaml              # Cấu hình tài nguyên, ngưỡng an toàn và siêu tham số
├── data/
│   ├── raw/                         # Dữ liệu cào thô ban đầu (TV1)
│   ├── processed/                   # Kho ngữ liệu sạch: corpus_dedup.parquet (600 bài)
│   └── fixtures/                    # Dữ liệu hợp đồng mẫu phục vụ kiểm thử độc lập
├── docs/                            # Toàn bộ tài liệu đặc tả kiến trúc học thuật & 5 vai trò
├── outputs/
│   ├── checkpoints/                 # Trọng số best_fusion_xgboost.json & fusion_scaler.json
│   └── reports/                     # Báo cáo thực nghiệm evaluation_metrics.json
├── sandbox/
│   └── test_random_articles.py      # Kịch bản kiểm thử 5 bài báo ngoài đời thực
├── src/
│   ├── crawlers/                    # Bộ cào tin tức & khử trùng lặp MinHash LSH (TV1)
│   ├── features/                    # Trích xuất Stylometrics (TV2), PhoBERT (TV3), Cross-Verifier (TV4)
│   ├── models/                      # DualStream PyTorch, Late Fusion XGBoost, SHAP Explainer
│   ├── api/                         # FastAPI REST Microservice (/api/v1/predict, /api/v1/explain)
│   └── utils/                       # Bộ công cụ tạo dữ liệu mẫu và trợ giúp
├── tests/
│   ├── test_components.py           # Bộ kiểm thử đơn vị các thành phần (TV2, TV3, TV4, TV5)
│   └── test_data_contracts.py       # Bộ kiểm thử hợp đồng dữ liệu chuẩn schema
├── app.py                           # Điểm chạy chính của Giao diện Trực quan Gradio Web UI
├── requirements.txt                 # Danh sách gói thư viện Python phụ thuộc
└── README.md                        # Tài liệu hướng dẫn này
```

---

## 7. LIÊN HỆ & BẢN QUYỀN
Dự án được xây dựng phục vụ nghiên cứu học thuật và ứng dụng thực tiễn trong khuôn khổ môn học Machine Learning / Khóa luận Tốt nghiệp CNTT. Phát hành theo giấy phép **MIT License**. Mọi đóng góp và phản hồi xin vui lòng tạo issue hoặc liên hệ với các thành viên trong nhóm dự án.
