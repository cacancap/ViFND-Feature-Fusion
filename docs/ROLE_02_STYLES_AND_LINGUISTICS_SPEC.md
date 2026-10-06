# ĐẶC TẢ VAI TRÒ & QUY TRÌNH KỸ THUẬT: THÀNH VIÊN 2 (TV2)
## KỸ SƯ NGÔN NGỮ & ĐẶC TRƯNG HÌNH THỨC (Linguistic & Stylometric Feature Engineer)

**Phạm vi phụ trách**: Tầng 3 (Tách từ ghép tiếng Việt) & Tầng 4B (Trích xuất 15 đặc trưng Hình thức Stylometrics).  
**Thư mục làm việc chính**: `src/features/`  
**Tập tin bàn giao chính**: 
1. `data/processed/corpus_tokenized.parquet`  
2. `data/processed/features/X_stylometric.npy` $(N \times 15)$  
3. `data/processed/features/stylometric_feature_names.json`  

---

## 1. MỤC TIÊU VÀ SỨ MỆNH KHOA HỌC
Sứ mệnh của TV2 là bẻ gãy vấn đề **"Trùng lặp từ vựng" (Vocabulary Overlap)**:
- Một bài báo chính thống điều tra vụ án lừa đảo và một bài viết giật gân bịa đặt đều chứa các từ vựng giống nhau (*công an, lừa đảo, tiền ảo*). Mô hình từ vựng thuần túy (TF-IDF) sẽ bị bẫy và dự đoán sai.
- TV2 chịu trách nhiệm lượng hóa **Phong cách biểu đạt (Writing Style)** của tin giả thành **15 chỉ số toán học chuẩn xác**. Báo chí chính thống luôn giữ văn phong trung tính, câu cú chuẩn mực; ngược lại, tin giả và giật gân luôn lạm dụng chữ hoa, dấu chấm than, ngôn từ kích động và tiêu đề bất cân xứng.

---

## 2. QUY TRÌNH KỸ THUẬT CỦA TV2

### Bước 1: Tách Từ Ghép Tiếng Việt Bằng PyVi (Word Segmentation)
- Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không tương ứng với dấu cách (ví dụ: *"học sinh"* là 1 từ ghép 2 âm tiết, khác hoàn toàn *"học"* và *"sinh"* đứng riêng lẻ).
- Sử dụng thư viện `pyvi.ViTokenizer`:
  ```python
  from pyvi import ViTokenizer
  tokenized_text = ViTokenizer.tokenize(clean_text)
  ```
  Kết quả sẽ nối các âm tiết thành từ ghép: *"học_sinh"*, *"chấn_động"*, *"bảo_hiểm_y_tế"*.
- Áp dụng tách từ ghép cho cả 2 trường `title` và `content`, lưu kết quả vào `data/processed/corpus_tokenized.parquet`.

### Bước 2: Xây Dựng Từ Điển Ngôn Ngữ Học Chuyên Miền (Domain Lexicons)
TV2 chịu trách nhiệm tạo và duy trì các tệp từ điển trong `src/features/lexicons/`:
1. **Từ điển Giật gân / Câu view (`clickbait_keywords.txt`)**: Gồm hơn 50 từ/cụm từ kích động cảm xúc mạnh:
   *Ví dụ: sốc, kinh hoàng, chấn động, ngã ngửa, rợn người, không thể tin nổi, bàng hoàng, động trời, cực sốc, lộ clip, xem ngay kẻo xóa...*
2. **Từ điển Cấp bách / Kêu gọi chia sẻ (`urgency_keywords.txt`)**: Gồm hơn 25 cụm từ thúc ép hành động:
   *Ví dụ: chia sẻ ngay, gấp, khẩn cấp, cứu người, chia sẻ gấp, bài thuốc thần kỳ, giấu kín, đừng dại dột...*
3. **Từ điển Xưng hô bình dân / Thiếu khách quan (`colloquial_pronouns.txt`)**: Gồm các đại từ chỉ xuất hiện trên mạng xã hội, không bao giờ dùng trong báo chí chính thống:
   *Ví dụ: bà con ơi, em thề, các bác ạ, cả nhà ơi, em nói thật, tụi mình...*

### Bước 3: Hiện Thực Hóa 15 Công Thức Trích Xuất Stylometrics
Với mỗi bài viết (gồm tiêu đề $T$ và thân bài $B$):

| STT | Tên Đặc Trưng | Công Thức Toán Học | Ý Nghĩa Phân Loại |
|:---:|---|---|---|
| 1 | `upper_ratio` | $\frac{\sum \mathbb{I}(c \text{ is upper})}{N_{\text{char}}}$ | Tin giả viết hoa vô tội vạ để tạo cảm giác la hét, khẩn cấp. |
| 2 | `exclamation_density` | $\frac{\text{Count}('!')}{N_{\text{word}} + 1}$ | Tin tức chuẩn mực không bao giờ dùng nhiều dấu chấm than (`!!!`). |
| 3 | `question_density` | $\frac{\text{Count}('?')}{N_{\text{word}} + 1}$ | Tiêu đề giật gân thường đặt câu hỏi nghi vấn để lôi kéo tò mò. |
| 4 | `punct_density` | $\frac{\sum \mathbb{I}(c \in \text{Punctuation})}{N_{\text{char}}}$ | Mật độ dấu câu hỗn loạn. |
| 5 | `title_body_ratio` | $\frac{N_{\text{word}}(T)}{\max(N_{\text{word}}(B), 1)}$ | Clickbait thường có tiêu đề rất dài nhưng thân bài rỗng tuếch. |
| 6 | `clickbait_keyword_hits` | $\sum_{w \in \mathcal{D}_{\text{clickbait}}} \text{Count}(w, D)$ | Số lần xuất hiện các từ kích động câu view. |
| 7 | `digit_in_title` | $\sum_{c \in T} \mathbb{I}(c \text{ is digit})$ | Tin lừa đảo thường đưa con số cụ thể vào tiêu đề (*100%, 500k, 7 ngày*). |
| 8 | `colon_dash_density` | $\frac{\text{Count}(':') + \text{Count}('-')}{N_{\text{sentence}} + 1}$ | Kỹ thuật ngắt nhịp kịch tính ở tiêu đề (*"CHẤN ĐỘNG: ..."*). |
| 9 | `quote_density` | $\frac{\text{Count}('"') + \text{Count}('”')}{N_{\text{word}} + 1}$ | Dấu ngoặc kép trích dẫn mập mờ, không rõ danh tính phát ngôn. |
| 10 | `avg_sentence_len` | $\frac{N_{\text{word}}}{N_{\text{sentence}} + 1}$ | Câu văn ngắn cụt ngủn hoặc câu dài lan man không ngữ pháp. |
| 11 | `urgency_density` | $\sum_{w \in \mathcal{D}_{\text{urgency}}} \text{Count}(w, D)$ | Thúc giục người đọc chia sẻ trước khi bài viết bị gỡ. |
| 12 | `pronoun_density` | $\sum_{w \in \mathcal{D}_{\text{pronoun}}} \text{Count}(w, D)$ | Đại từ nhân xưng phi chuẩn mực báo chí (*bà con, em thề*). |
| 13 | `teencode_ratio` | $\frac{\sum_{w \in D} \mathbb{I}(w \in \mathcal{D}_{\text{teencode}})}{N_{\text{word}} + 1}$ | Sử dụng tiếng lóng, viết tắt (*ko, dc, mn, j*). |
| 14 | `title_body_jaccard` | $\frac{\|V(T) \cap V(B)\|}{\|V(T) \cup V(B)\|}$ | Độ lệch giữa tiêu đề và thân bài (Treo đầu dê bán thịt chó). |
| 15 | `sentiment_extremity` | $\|P_{\text{pos}} - P_{\text{neg}}\|$ | Mức độ cảm xúc cực đoan (quá phấn khích hoặc quá hoảng sợ). |

### Bước 4: Chuẩn Hóa Đặc Trưng (Feature Scaling) & Xuất File
- Ma trận 15 đặc trưng thô sẽ được chuẩn hóa Z-score bằng `sklearn.preprocessing.StandardScaler` ($\mu = 0, \sigma = 1$).
- Xuất kết quả sang tệp nhị phân: `data/processed/features/X_stylometric.npy`.
- Xuất danh sách tên đặc trưng: `data/processed/features/stylometric_feature_names.json`.

---

## 3. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE - DOD)
- [ ] Vượt qua test tự động: `pytest tests/test_data_contracts.py::test_contract_p2`.
- [ ] Ma trận `X_stylometric.npy` có kích thước chính xác $(N, 15)$, kiểu dữ liệu `np.float32`.
- [ ] Tuyệt đối không chứa giá trị `NaN`, `Inf` hoặc giá trị null.
- [ ] Tốc độ trích xuất đạt tối thiểu **500 bài viết / giây trên CPU**.
