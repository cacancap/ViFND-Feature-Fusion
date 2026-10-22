# TÀI LIỆU ĐẶC TẢ KIẾN TRÚC HỆ THỐNG VÀ QUY CHUẨN VAI TRÒ NHÓM 5 NGƯỜI
## DỰ ÁN: ViFND-Feature-Fusion (Phát Hiện Tin Giả Tiếng Việt Qua Hợp Nhất Đặc Trưng Ba Luồng & XAI)

**Cập nhật lần cuối**: Tháng 10/2026  
**Mục tiêu dự án**: Xây dựng hệ thống học máy phát hiện tin giả, tin giật gân và tin đồn vô căn cứ trên báo mạng và mạng xã hội tiếng Việt bằng cách kết hợp ba luồng đặc trưng độc lập:
1. Biểu diễn ngữ nghĩa sâu (PhoBERT 768 chiều).
2. Chỉ số phong cách hình thức (15 công thức Stylometrics).
3. Tầng kiểm chứng chéo đa nguồn (Cross-Source Corroboration Engine Hybrid 4 chiều).
Tích hợp khả năng minh giải quyết định bằng SHAP TreeExplainer và triển khai ứng dụng Web tương tác Gradio + REST API FastAPI.

---

## 1. CƠ SỞ LÝ LUẬN & ĐỊNH NGHĨA KHOA HỌC VỀ "TIN GIẢ"

Trong giới nghiên cứu Khoa học Dữ liệu và Xử lý Ngôn ngữ Tự nhiên (NLP), khái niệm "Tin giả" (Fake News) thường bị phê bình vì quá mơ hồ nếu không có khung lý thuyết chuẩn tắc. Dự án ViFND chuẩn hóa bài toán dựa trên **Khung lý thuyết Rối loạn Thông tin (Information Disorder Framework)** do Claire Wardle & Hossein Derakhshan (Harvard Shorenstein Center & Hội đồng Châu Âu) đề xuất:

```
                                ┌──────────────────────────────────────────────┐
                                │       INFORMATION DISORDER FRAMEWORK         │
                                └──────────────────────────────────────────────┘
                                                       │
                 ┌─────────────────────────────────────┼─────────────────────────────────────┐
                 ▼                                     ▼                                     ▼
   ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
   │      MISINFORMATION       │         │      DISINFORMATION       │         │       MALINFORMATION      │
   │ (Sai sự thật, KHÔNG ác ý) │         │ (Sai sự thật, CÓ ÁC Ý)    │         │ (Có thật, DÙNG ĐỂ GÂY HẠI)│
   ├───────────────────────────┤         ├───────────────────────────┤         ├───────────────────────────┤
   │ - Tin đồn thiếu kiểm chứng│         │ - Bịa đặt 100% (Fabr.)    │         │ - Rò rỉ đời tư bôi nhọ    │
   │ - Tiêu đề giật gân một    │         │ - Giả mạo cơ quan chức    │         │ - Sử dụng sai ngữ cảnh    │
   │   đằng, nội dung một nẻo  │         │   năng (Imposter)         │         │   nhằm triệt hạ uy tín    │
   │   (Clickbait Incongruence)│         │ - Cắt ghép xuyên tạc      │         │                           │
   └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
```

### Tại sao mô hình thuần PhoBERT + Stylometrics là chưa đủ?
- **PhoBERT** chỉ mã hóa ngữ cảnh câu chữ và phân bố từ ngữ.
- **Stylometrics** chỉ đo lường mức độ giật gân, la hét cảm xúc (dấu chấm than, viết hoa, từ kích động).
- 👉 **Hạn chế thực tế**: Một bài viết có thể viết bằng giọng điệu vô cùng đàng hoàng, chuẩn mực của báo chí (Stylometrics thấp) nhưng nội dung lại hoàn toàn bịa đặt (Fabrication) một sự kiện không có thật. Nếu không có **tri thức kiểm chứng đối chiếu (Veracity / Corroboration)**, mô hình thuần phong cách sẽ bị qua mặt.

### Giải pháp của ViFND: Kiến trúc Ba Luồng (Tri-Stream Late Fusion)
Hệ thống kết hợp đồng thời 3 góc nhìn:
1. **Góc nhìn Ngữ nghĩa (Semantic)**: Bài viết đang nói về chủ đề gì, cấu trúc câu mạch lạc hay rời rạc?
2. **Góc nhìn Văn phong (Stylistic)**: Tác giả có đang cố tình thao túng cảm xúc, kích động la hét, thúc ép chia sẻ (Clickbait cues) hay không?
3. **Góc nhìn Kiểm chứng Thực tế (Veracity / Factuality)**: Sự kiện được tuyên bố trong bài viết có được các cơ quan báo chí chính thống uy tín (VnExpress, Tuổi Trẻ, VietnamNet, VTV, Thông tấn xã) xác nhận hay không?

---

## 2. TỔNG QUAN KIẾN TRÚC HỆ THỐNG 7 TẦNG (7-TIER ARCHITECTURE)

```
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: THU THẬP ĐA NGUỒN & TRUY XUẤT XUẤT XỨ (TV1)                   │
│ - RSS Báo chính thống: VnExpress, Tuổi Trẻ, VietnamNet (Nhãn 0)        │
│ - Blog / Diễn đàn giật gân, Cổng tingia.gov.vn, Bộ VFND (Nhãn 1)       │
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
└───────────────────┬────────────────┬───────────────────┬───────────────┘
                    │                │                   │
                    ▼                ▼                   ▼
┌───────────────────────┐ ┌─────────────────────┐ ┌───────────────────────┐
│ TẦNG 4A: NGỮ NGHĨA SÂU│ │ TẦNG 4B: HÌNH THỨC  │ │ TẦNG 4C: KIỂM CHỨNG   │
│ (TV3 - PhoBERT)       │ │ (TV2 - Stylometrics)│ │ CHÉO ĐA NGUỒN (MỚI)   │
│ - vinai/phobert-base-v2│ - 15 chỉ số toán học  │ - CrossSourceVerifier │
│ - Head-Tail Truncation│ - Mật độ !, ?, chữ hoa│ - Local TF-IDF/BM25   │
│ - Attention Mean Pool │ - Từ khóa giật gân    │ - Google News RSS fall│
│ - Chuẩn hóa L2        │ - StandardScaler      │ - 4 chỉ số kiểm chứng │
│ - X_semantic (N x 768)│ - X_stylo (N x 15)    │ - X_veri (N x 4)      │
└───────────┬───────────┘ └──────────┬──────────┘ └──────────┬────────────┘
            └────────────────────────┼───────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5: HỢP NHẤT ĐẶC TRƯNG MUỘN (LATE FUSION) & MÔ HÌNH HÓA (TV4)      │
│ - Mô hình A: Ghép vector 787 chiều + XGBoost / CatBoost                │
│ - Mô hình B: Mạng nơ-ron đa luồng PyTorch (Dual/Multi-Stream NN)       │
│ - Tối ưu siêu tham số: Stratified 5-Fold CV + Optuna                   │
│ - Xuất checkpoint: best_fusion_xgboost.json, best_fusion_pytorch.pt    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 6: TRÍ TUỆ NHÂN TẠO GIẢI THÍCH ĐƯỢC (XAI) (TV4)                   │
│ - SHAP TreeExplainer bóc tách % rủi ro từng đặc trưng (Văn phong +     │
│   Kiểm chứng sự kiện: 'Phạt +35% vì 0 có nguồn chính thống xác nhận') │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 7: DEMO WEB & GIAO DIỆN TƯƠNG TÁC (TV5)                           │
│ - Web Demo Gradio: Dán URL/nhập text, đồng hồ đo rủi ro, biểu đồ SHAP  │
│   và hiển thị danh sách bài báo đối chứng từ báo chí chính thống       │
│ - Dịch vụ REST API FastAPI: /health, /api/v1/predict, /api/v1/explain  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. BẢNG TỔNG HỢP HỢP ĐỒNG GIAO TIẾP DỮ LIỆU (DATA CONTRACTS MATRIX)

| Mã Hợp Đồng | Bên Giao | Bên Nhận | File Bàn Giao Thật | File Mẫu Ngày 1 (Fixture) | Ràng Buộc Kỹ Thuật Bắt Buộc |
|---|:---:|:---:|---|---|---|
| **Contract 1** | **TV1** | **TV2, TV3** | `data/processed/corpus_dedup.parquet` | `data/fixtures/sample_corpus.parquet` | Bảng 6 cột chuẩn (`doc_id`, `title`, `content`, `source`, `url`, `label`). Không chứa giá trị null. |
| **Contract 2** | **TV2** | **TV4** | `data/processed/features/X_stylometric.npy` | `data/fixtures/sample_X_stylometric.npy` | Tensor 2D kích thước $(N, 15)$, kiểu `float32`, chuẩn hóa `StandardScaler` ($\mu=0, \sigma=1$). Không chứa NaN/Inf. |
| **Contract 3** | **TV3** | **TV4** | `data/processed/features/X_semantic.npy` | `data/fixtures/sample_X_semantic.npy` | Tensor 2D kích thước $(N, 768)$, kiểu `float32`, chuẩn hóa vector đơn vị L2 ($\|\mathbf{z}\|_2 = 1.0$). |
| **Contract 4** | **Verifier**| **TV4** | `data/processed/features/X_verification.npy` | `data/fixtures/sample_X_verification.npy` | Tensor 2D kích thước $(N, 4)$, kiểu `float32`, giá trị trong khoảng $[0.0, 1.0]$. |
| **Contract 5** | **TV4** | **TV5** | `src/models/fusion_classifier.py` | `data/fixtures/sample_prediction_output.json` | Hàm `predict_with_shap()` trả về Dict đúng cấu trúc: `prediction`, `probability_fake`, `top_shap_factors`, `evidence_articles`. |

---

## 4. QUY TRÌNH KIỂM THỬ TỰ ĐỘNG CHẤT LƯỢNG (AUTOMATED QA)

Chạy kiểm thử toàn bộ hệ thống bằng lệnh:
```bash
python -m pytest tests/
```
Bộ kiểm thử tự động kiểm tra:
1. `test_components.py`: Đảm bảo 100% các module TV2, TV3, TV4 (CrossSourceVerifier, DualStream NN, SHAP) và TV5 (FastAPI) hoạt động chính xác.
2. `test_data_contracts.py`: Đảm bảo toàn bộ 5 hợp đồng dữ liệu giữa các thành viên tuân thủ nghiêm ngặt schema và định dạng tensor.
