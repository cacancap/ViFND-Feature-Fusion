# ĐẶC TẢ VAI TRÒ & QUY TRÌNH KỸ THUẬT: THÀNH VIÊN 5 (TV5)
## KỸ SƯ MLOPS, GIAO DIỆN WEB & ĐIỀU PHỐI BẢO VỆ (Full-Stack MLOps, UI Demo & Defense Lead)

**Phạm vi phụ trách**: Triển khai ứng dụng Web tương tác (Gradio), dịch vụ API (FastAPI), đóng gói Docker, và điều phối Slide thuyết trình bảo vệ 15 phút.  
**Thư mục làm việc chính**: `src/ui/`, `src/api/`, thư mục gốc (`app.py`, `Dockerfile`)  
**Tập tin bàn giao chính**: 
1. Ứng dụng Web chính: `app.py` (Gradio)  
2. REST API Microservice: `src/api/server.py` (FastAPI)  
3. Đóng gói môi trường: `Dockerfile` & `docker-compose.yml`  
4. Bộ Slide thuyết trình bảo vệ: `docs/oral_defense_slides.pptx` (hoặc PDF)  

---

## 1. NGUYÊN TẮC VÀNG CỦA TV5: TRÁNH LẰNG NHẰNG, TẬP TRUNG CỐT LÕI (KISS PRINCIPLE)
Nhiệm vụ của TV5 là tạo ra **điểm nhấn thị giác mạnh nhất cho toàn bộ đề tài** trước Hội đồng chấm thi ("Professor Impact Factor").  
Để tránh lãng phí thời gian vào những việc không cộng điểm trong môn Khai phá Dữ liệu, TV5 **TUYỆT ĐỐI TUÂN THỦ CÁC QUY TẮC SAU**:
- ❌ **KHÔNG LÀM**: Hệ thống đăng nhập, đăng ký, xác thực JWT, cơ sở dữ liệu lưu tài khoản người dùng, bảng điều khiển admin quản trị người dùng. Những tính năng này thuộc môn Lập trình Web, không có giá trị học thuật trong môn Data Mining.
- ✔️ **TẬP TRUNG 100% VÀO**: Trải nghiệm phân tích tin giả tức thì (Real-Time Inference), tính năng tự động cào tin từ link URL, và **trực quan hóa biểu đồ SHAP Waterfall** giúp giảng viên nhìn thấy rõ lý do mô hình đưa ra cảnh báo.

---

## 2. ĐẶC TẢ 3 KHỐI CHỨC NĂNG DUY NHẤT CỦA GIAO DIỆN WEB (GRADIO)

Giao diện Web được xây dựng bằng **Gradio** (chạy tại cổng `http://localhost:7860`), được thiết kế tinh gọn gồm 3 khối chức năng:

```
┌────────────────────────────────────────────────────────────────────────┐
│        HỆ THỐNG PHÁT HIỆN TIN GIẢ TIẾNG VIỆT QUA HỢP NHẤT ĐẶC TRƯNG    │
├────────────────────────────────────────────────────────────────────────┤
│ KHỐI 1: ĐẦU VÀO BÀI BÁO (INPUT MODULE)                                 │
│ ┌───────────────────────────────────┐ ┌──────────────────────────────┐ │
│ │ Tab 1: Dán đường dẫn URL bài báo  │ │ Tab 2: Nhập văn bản tùy ý    │ │
│ │ [https://vnexpress.net/...   ] 🔍 │ │ [Tiêu đề: ...             ]  │ │
│ │ (Hệ thống tự cào và trích xuất)   │ │ [Nội dung: ...            ]  │ │
│ └───────────────────────────────────┘ └──────────────────────────────┘ │
│                                [ KIỂM TRA TIN TỨC ]                    │
├────────────────────────────────────────────────────────────────────────┤
│ KHỐI 2: KẾT QUẢ DỰ ĐOÁN & ĐỒNG HỒ ĐO RỦI RO (PREDICTION MODULE)       │
│ • Nhãn cảnh báo: [ 🚨 TIN CÓ DẤU HIỆU GIẢ MẠO / GIẬT GÂN ] (Màu đỏ)    │
│ • Độ tin cậy (Confidence Score): 94.2%  |  Mức độ: NGUY HIỂM CAO      │
│ • Thời gian xử lý: 0.85 giây                                           │
├────────────────────────────────────────────────────────────────────────┤
│ KHỐI 3: MINH GIẢI QUYẾT ĐỊNH XAI (SHAP EXPLANATION - ĐIỂM ĂN TIỀN NHẤT)│
│ • Biểu đồ thanh ngang SHAP Waterfall Plot trực quan:                   │
│   [+] exclamation_density (+0.48) : Mật độ dấu chấm than (!) dày đặc   │
│   [+] clickbait_keywords  (+0.35) : Chứa từ kích động "CHẤN ĐỘNG"      │
│   [-] phobert_semantics   (-0.12) : Ngữ nghĩa nội dung bài viết        │
│ • 3 gạch đầu dòng tóm tắt cảnh báo bằng tiếng Việt tự nhiên.           │
└────────────────────────────────────────────────────────────────────────┘
```

### Chi tiết kỹ thuật từng khối:
1. **Khối 1: Module Đầu vào (Input Module)**:
   - *Tab A (Dán URL)*: Người dùng chỉ cần dán link báo bất kỳ (VnExpress, Tuổi Trẻ, blog mạng...). TV5 tích hợp hàm cào nhanh (sử dụng logic của TV1) để tự động trích xuất tiêu đề và thân bài trong $<0.5$ giây.
   - *Tab B (Nhập tự do)*: Cho phép ban giám khảo tự gõ tiêu đề và nội dung để test các trường hợp biên.
2. **Khối 2: Module Dự đoán (Prediction Module)**:
   - Gọi hàm `predict_with_shap()` từ TV4.
   - Hiển thị nhãn nổi bật: `TIN XÁC THỰC` (Màu xanh lá) hoặc `TIN GIẢ / GIẬT GÂN` (Màu đỏ/cam).
   - Hiển thị thanh đo mức độ tự tin (Confidence bar: 0 - 100%).
3. **Khối 3: Module Trực quan hóa XAI (Explainability Module)**:
   - Đây là module giúp nhóm ghi điểm tuyệt đối. Thay vì trả về một con số khô khan, giao diện vẽ trực tiếp biểu đồ SHAP thanh ngang (Horizontal Bar Chart) hiển thị rõ các yếu tố nào làm tăng nguy cơ tin giả và yếu tố nào làm giảm nguy cơ.
   - Hiển thị phần tóm tắt lý do bằng tiếng Việt thân thiện với người dùng thông thường.

---

## 3. DỊCH VỤ REST API (FASTAPI)
Bên cạnh giao diện Gradio, TV5 xây dựng dịch vụ REST API chuẩn mực tại `src/api/server.py` với tài liệu Swagger tự động tại `/docs`:
- `POST /api/v1/predict`: Nhận `{ "title": "...", "content": "..." }`, trả về nhãn và xác suất.
- `POST /api/v1/explain`: Nhận bài viết, trả về danh sách các giá trị đóng góp SHAP.
- `GET /health`: Kiểm tra trạng thái sẵn sàng của hệ thống.

---

## 4. KỊCH BẢN THUYẾT TRÌNH BẢO VỆ 15 PHÚT (ORAL DEFENSE ALLOCATION)
TV5 là người điều phối tổng thể bài thuyết trình trước Hội đồng phản biện. Phân bổ thời gian chuẩn xác:

| Khung Thời Gian | Người Trình Bày | Nội Dung Trọng Tâm |
|---|---|---|
| **00:00 – 03:00** | **TV1 (Data Eng)** | Vấn nạn tin giả tiếng Việt $\rightarrow$ Quy trình cào đa nguồn $\rightarrow$ **Điểm nhấn MinHash LSH** loại bỏ hơn 300 bài báo sao chép trùng lặp, chống rò rỉ dữ liệu. |
| **03:00 – 06:00** | **TV2 (Stylometrics)** | Tại sao mô hình từ vựng truyền thống thất bại $\rightarrow$ Giải pháp tách từ ghép PyVi $\rightarrow$ **15 công thức đặc trưng hình thức** phong cách giật gân. |
| **06:00 – 09:00** | **TV3 (PhoBERT)** | Cơ chế nhúng sâu PhoBERT $\rightarrow$ Kỹ thuật cắt tỉa **Head-Tail Truncation** giải quyết giới hạn 256 token $\rightarrow$ Trích xuất vector 768 chiều trên Colab T4. |
| **09:00 – 12:00** | **TV4 (Modeling & XAI)** | Kiến trúc **Late Fusion PyTorch đa nhánh** $\rightarrow$ Kết quả thực nghiệm nhảy vọt (+20.87% F1) $\rightarrow$ **Minh giải SHAP** chứng minh tầm quan trọng của đặc trưng hình thức. |
| **12:00 – 15:00** | **TV5 (UI Demo)** | **DEMO TRỰC TIẾP TRÊN MÁY CHIẾU**: Dán 1 link bài báo thật và 1 link bài giật gân, phân tích trong 1 giây, chỉ cho hội đồng thấy biểu đồ SHAP Waterfall. Kết luận đề tài. |

---

## 5. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE - DOD)
- [ ] Ứng dụng Web `app.py` chạy ổn định trên máy cá nhân (`python app.py` mở cổng 7860).
- [ ] Thời gian phản hồi từ lúc bấm nút đến lúc hiển thị kết quả và biểu đồ SHAP **dưới 1.5 giây**.
- [ ] Có đầy đủ 2 chế độ: Nhập URL tự động và Nhập văn bản thủ công.
- [ ] Đóng gói thành công `Dockerfile` có thể chạy trên bất kỳ máy tính nào mà không bị lỗi thiếu thư viện.
- [ ] Bộ slide thuyết trình 15 phút được hoàn thiện trước ngày bảo vệ tối thiểu 1 tuần để nhóm diễn tập.
