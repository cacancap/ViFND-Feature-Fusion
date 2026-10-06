# ĐẶC TẢ VAI TRÒ & QUY TRÌNH KỸ THUẬT: THÀNH VIÊN 1 (TV1)
## KỸ SƯ DỮ LIỆU & THU THẬP WEB (Data Engineering & Web Harvesting Lead)

**Phạm vi phụ trách**: Toàn bộ Tầng 1 (Thu thập đa nguồn) & Tầng 2 (Làm sạch thô & Khử trùng lặp LSH).  
**Thư mục làm việc chính**: `src/crawlers/`, `data/raw/`, `data/processed/`  
**Tập tin bàn giao chính**: `data/processed/corpus_dedup.parquet`  

---

## 1. MỤC TIÊU VÀ SỨ MỆNH KHOA HỌC
Trong môn học Khai phá Dữ liệu, chất lượng của dữ liệu đầu vào quyết định 80% thành công của mô hình ("Garbage in, Garbage out"). Sứ mệnh của TV1 là:
1. Xây dựng bộ dữ liệu tiếng Việt đa nguồn cân bằng, có tính đại diện cao cho cả tin chính thống và tin đồn giả mạo.
2. Xây dựng pipeline tiền xử lý kỹ thuật cao: Bóc tách rác HTML, chuẩn hóa Unicode NFC, và đặc biệt là **khử trùng lặp nội dung báo chí bằng thuật toán MinHash LSH** để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) khi chia tập huấn luyện/kiểm thử.

---

## 2. QUY TRÌNH KỸ THUẬT 4 BƯỚC CỦA TV1

### Bước 1: Thu Thập Đa Nguồn & Ghi Vết Xuất Xứ (Provenance Tracking)
- **Nguồn Tin Thật (Nhãn 0 - Real / Verified)**:
  - Cào tự động qua kênh RSS chuẩn của các tòa soạn chính thống:
    - *VnExpress*: `https://vnexpress.net/rss/thoi-su.rss`, `kinh-doanh.rss`, `the-gioi.rss`, `khoa-hoc.rss`
    - *Tuổi Trẻ*: `https://tuoitre.vn/rss/thoi-su.rss`, `the-gioi.rss`
    - *VietnamNet*: `https://vietnamnet.vn/rss/thoi-su.rss`
  - *Cào toàn văn (Full-text extraction)*: Sau khi đọc RSS lấy được link URL, gửi request bóc tách toàn bộ các đoạn văn thân bài:
    - VnExpress: `<article class="fck_detail"> p.Normal`
    - Tuổi Trẻ: `div[class*="content"] p`
  - *Quy tắc lịch sự (Polite Crawling)*: Bắt buộc đặt `time.sleep(1.0)` giữa các request và khai báo User-Agent trình duyệt chuẩn để không bị chặn IP.
- **Nguồn Tin Giả / Tin Giật Gân (Nhãn 1 - Fake / Sensational Clickbait)**:
  - Nguồn 1: Thu thập từ các trang tin tức giải trí giật gân, blog mạng xã hội unverified.
  - Nguồn 2: Tải và tích hợp bộ dữ liệu chuẩn **VFND (Vietnamese Fake News Dataset)** đã được cộng đồng nghiên cứu kiểm duyệt nhãn.
  - Mục tiêu quy mô: Thu thập tối thiểu 4,500 – 5,000 bài viết thô (tỷ lệ cân bằng khoảng 50% Real / 50% Fake).

### Bước 2: Bóc Tách HTML & Làm Sạch Văn Bản Thô
- Loại bỏ mã HTML thừa bằng `BeautifulSoup(html, "lxml")`.
- Loại bỏ triệt để:
  - Các khối quảng cáo, liên kết đọc thêm ("Tin liên quan:", "Xem thêm:", "Video:", "Ảnh:").
  - Caption chú thích ảnh, tên tác giả/nguồn tin ở cuối bài (để mô hình không học vẹt tên tác giả).
- **Chuẩn hóa bảng mã**:
  - Tiếng Việt rất dễ bị xung đột giữa Unicode dựng sẵn (NFC) và Unicode tổ hợp (NFD).
  - Sử dụng module `unicodedata`:
    ```python
    import unicodedata
    clean_text = unicodedata.normalize('NFC', text)
    ```
- Xóa khoảng trắng thừa, ký tự xuống dòng liên tiếp (`\r\n\t` $\rightarrow$ ` `).

### Bước 3: Khử Trùng Lặp Cận Âm Bằng MinHash & Locality-Sensitive Hashing (LSH)
*Đây là điểm nhấn học thuật quan trọng nhất trong báo cáo của TV1.*

1. **Vấn đề thực tế**: Các báo mạng tại Việt Nam liên tục sao chép hoặc trích dẫn thông cáo báo chí của nhau (syndication). Nếu để nguyên, 1 bài viết có thể xuất hiện 5 lần ở các nguồn khác nhau. Khi chia train/test ngẫu nhiên, bài viết này sẽ xuất hiện ở cả tập train và tập test $\rightarrow$ Điểm F1 ảo lên tới 99% nhưng mô hình vô dụng ngoài thực tế.
2. **Quy trình thuật toán**:
   - **Shingling**: Chuyển đổi văn bản thành tập hợp các character 5-shingles (chuỗi 5 ký tự liên tiếp).
   - **MinHash**: Sử dụng thư viện `datasketch.MinHash` với $K = 128$ hàm băm xác suất độc lập để tạo vector chữ ký số (MinHash signature) xấp xỉ độ tương đồng Jaccard.
   - **LSH Indexing**: Phân chia 128 hàm băm thành $b = 32$ băng (bands), mỗi băng $r = 4$ hàng ($32 \times 4 = 128$).
   - Thiết lập ngưỡng tương đồng: $\theta = 0.85$.
   - **Chính sách giữ lại (Retention Policy)**: Với mỗi cụm bài viết có độ tương đồng Jaccard $\ge 0.85$:
     - Giữ lại bản ghi có thời gian xuất bản sớm nhất (bài gốc).
     - Loại bỏ các bài sao chép phía sau.
3. **Kết quả đo lường**: Ghi lại nhật ký: Số bài ban đầu $\rightarrow$ Số bài bị loại bỏ bởi LSH $\rightarrow$ Số bài còn lại duy nhất.

### Bước 4: Kiểm Định Chất Lượng (Data Quality Audit) & Xuất File Parquet
- Lọc bỏ các bài viết rác:
  - Số lượng từ thân bài $< 30$ từ (bài quá ngắn không đủ thông tin).
  - Bài viết chỉ toàn link ảnh hoặc video không có chữ.
  - Bài báo lỗi kết nối hoặc chuyển hướng 404.
- Đóng gói dữ liệu đầu ra sang tệp `data/processed/corpus_dedup.parquet` (sử dụng thư viện `pyarrow` với chuẩn nén Snappy).

---

## 3. ĐẶC TẢ HỢP ĐỒNG ĐẦU RA (OUTPUT CONTRACT SPECIFICATION)

Tệp đầu ra: `data/processed/corpus_dedup.parquet`  
Schema kỹ thuật bắt buộc:

| Tên Cột (Column) | Kiểu Dữ Liệu | Ràng Buộc | Mô Tả & Ví Dụ |
|---|---|:---:|---|
| `doc_id` | `string` | Khóa chính, duy nhất | Mã định danh bài viết, ví dụ: `"VNNEWS_000123"` |
| `title` | `string` | Không null, độ dài > 10 | Tiêu đề bài viết sau khi làm sạch |
| `content` | `string` | Không null, độ dài > 50 | Toàn bộ nội dung thân bài viết đã bóc tách HTML |
| `source` | `string` | Không null | Tên nguồn hoặc tên miền, ví dụ: `"VnExpress_ThoiSu"` |
| `url` | `string` | Không null | Link gốc của bài viết phục vụ kiểm chứng |
| `crawled_at` | `string` | Định dạng ISO 8601 | Thời điểm thu thập, ví dụ: `"2026-10-06T10:00:00Z"` |
| `label` | `int8` | 0 hoặc 1 | Nhãn phân loại: `0` = Tin thật, `1` = Tin giả / Giật gân |

---

## 4. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE - DOD)
TV1 được coi là hoàn thành xuất sắc nhiệm vụ khi thỏa mãn toàn bộ các tiêu chí sau:
- [ ] File `data/processed/corpus_dedup.parquet` tồn tại và vượt qua bài test tự động: `pytest tests/test_data_contracts.py::test_contract_p1`.
- [ ] Số lượng mẫu thực tế đạt tối thiểu **4,000 bài viết duy nhất** (tỷ lệ cân bằng khoảng 45% - 55% giữa nhãn 0 và 1).
- [ ] 100% không còn cặp bài viết nào có độ tương đồng Jaccard $\ge 0.85$ (đã lọc triệt để qua MinHash LSH).
- [ ] Không có bất kỳ ô dữ liệu nào bị `NaN`, `None` hoặc chuỗi rỗng ở các cột `title` và `content`.
- [ ] Có tài liệu thống kê ngắn gọn (EDA summary) mô tả phân bố độ dài từ, số lượng bài theo từng nguồn báo chí.
