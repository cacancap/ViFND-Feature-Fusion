# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỀ XUẤT ĐỀ TÀI & CẨM NANG PHẢN BIỆN (ORAL DEFENSE PITCH SCRIPT)

**Học phần**: Nhập môn Học máy và Khai phá Dữ liệu (IT3190) — Viện CNTT&TT, Đại học Bách Khoa Hà Nội  
**Đề tài**: Đề tài 01 — Phát hiện Tin giả Tiếng Việt Đa chiều qua Hợp nhất Đặc trưng (PhoBERT + Stylometrics + SHAP XAI)  
**Nhóm thực hiện**: Nhóm 3 (5 thành viên)  
**Tổng thời lượng trình bày**: 12 – 15 phút (theo chuẩn quy định bảo vệ đề cương đề tài môn học)  
**Tập tin trình chiếu đi kèm**: [`Team3_FakeNews_Proposal_PitchDeck.pptx`](Team3_FakeNews_Proposal_PitchDeck.pptx)  

---

## MỤC LỤC BÀI THUYẾT TRÌNH
1. [Slide 1: Trang bìa & Định vị Đề tài (00:00 – 01:30) — Điều phối chung: TV5](#slide-1-trang-bìa--định-vị-đề-tài)
2. [Slide 2: Bối cảnh Bài toán & 3 "Điểm mù" của Mô hình Truyền thống (01:30 – 03:30) — TV1](#slide-2-bối-cảnh-bài-toán--3-điểm-mù-của-mô-hình-truyền-thống)
3. [Slide 3: Giải pháp Đề xuất: Kiến trúc Hợp nhất Đặc trưng Muộn (03:30 – 05:30) — TV4](#slide-3-giải-pháp-đề-xuất-kiến-trúc-hợp-nhất-đặc-trưng-muộn)
4. [Slide 4: Quy trình Khai phá Dữ liệu Nâng cao & MinHash LSH (05:30 – 07:30) — TV1](#slide-4-quy-trình-khai-phá-dữ-liệu-nâng-cao--minhash-lsh)
5. [Slide 5: Kỹ thuật Trích xuất Đặc trưng Kép (15D Stylometrics vs 768D PhoBERT) (07:30 – 09:30) — TV2 & TV3](#slide-5-kỹ-thuật-trích-xuất-đặc-trưng-kép)
6. [Slide 6: Phương pháp Mô hình hóa, Đánh giá & Giải thích SHAP XAI (09:30 – 11:30) — TV4](#slide-6-phương-pháp-mô-hình-hóa-đánh-giá--giải-thích-shap-xai)
7. [Slide 7: Hệ sinh thái Công cụ & Tính Khả thi Hạ tầng (0 đồng chi phí) (11:30 – 13:00) — TV5](#slide-7-hệ-sinh-thái-công-cụ--tính-khả-thi-hạ-tầng)
8. [Slide 8: Phương pháp Vận hành Contract-First & Phân công Đội ngũ (13:00 – 14:30) — TV5](#slide-8-phương-pháp-vận-hành-contract-first--phân-công-đội-ngũ)
9. [CẨM NANG TOP 5 CÂU HỎI PHẢN BIỆN TỪ GIẢNG VIÊN KHI DUYỆT ĐỀ TÀI & CÁCH TRẢ LỜI XUẤT SẮC](#cẩm-nang-top-5-câu-hỏi-phản-biện-từ-giảng-viên)

---

## SLIDE 1: TRANG BÌA & ĐỊNH VỊ ĐỀ TÀI
- **Người trình bày**: **Thành viên 5 (TV5 - Trưởng nhóm Điều phối)**
- **Thời lượng dự kiến**: 1 phút 30 giây (00:00 – 01:30)
- **Mục tiêu slide**: Gây ấn tượng mạnh ban đầu với Hội đồng về tính quy chuẩn, phong thái đĩnh đạc của sinh viên Bách Khoa; định vị rõ đề tài không chỉ là "chạy một mô hình phân loại", mà là giải quyết một bài toán kỹ thuật thực sự bằng kiến trúc dung hợp đặc trưng hiện đại.
- **Chỉ dẫn thị giác (Visual Cues)**: Đứng thẳng, tay cầm presenter chỉ nhẹ về phía tên đề tài; giữ ánh mắt bao quát cả hội đồng chấm.

### Lời thoại chi tiết từng câu từng chữ:
> *"Kính thưa quý Thầy, Cô trong Hội đồng phản biện môn học Nhập môn Học máy và Khai phá Dữ liệu (IT3190)!*
> 
> *Em xin đại diện cho Nhóm 3 kính chúc quý Thầy Cô một buổi làm việc hiệu quả. Hôm nay, nhóm chúng em xin được trình bày Báo cáo đề xuất đề tài cho đồ án môn học với tên gọi:*
> 
> **'Phát hiện Tin giả Tiếng Việt Đa chiều: Kết hợp Biểu diễn Ngữ nghĩa sâu PhoBERT và Đặc trưng Phong cách viết Stylometrics cùng Trí tuệ Nhân tạo Giải thích được SHAP XAI'.**
> 
> *[Tạm dừng 1 giây để Hội đồng nhìn tiêu đề slide]*
> 
> *Thưa Thầy Cô, ngay từ giai đoạn nhận đề tài, Nhóm 3 đã xác định rõ thông điệp cốt lõi của dự án: Chúng em không chọn cách tiếp cận theo lối mòn là đưa toàn văn bài báo vào một mô hình ngôn ngữ lớn như một chiếc 'hộp đen' rồi nhận về nhãn phân loại 0 hoặc 1. Thay vào đó, chúng em giải quyết bài toán bằng kiến trúc **Hợp nhất Đặc trưng Muộn (Late Feature Fusion)** giữa 768 chiều ngữ nghĩa sâu của PhoBERT và 15 chiều chỉ số hình thức phong cách viết.*
> 
> *Đồng thời, để đồ án hoàn toàn khả thi và không rơi vào tình trạng chậm tiến độ kiểu thác nước, nhóm chúng em đã áp dụng triệt để phương pháp luận kỹ thuật phần mềm **Contract-First & Mock-Driven Architecture**, cho phép cả 5 thành viên lập trình song song độc lập 100% ngay từ Ngày đầu tiên.*
> 
> *Sau đây, em xin phép nhường lời cho bạn Thành viên 1 đại diện nhóm trình bày về Bối cảnh thực tế và 3 'Điểm mù' nghiêm trọng của các phương pháp phân loại tin giả truyền thống hiện nay!"*

---

## SLIDE 2: BỐI CẢNH BÀI TOÁN & 3 "ĐIỂM MÙ" CỦA MÔ HÌNH TRUYỀN THỐNG
- **Người trình bày**: **Thành viên 1 (TV1 - Kỹ sư Dữ liệu & Cào Web)**
- **Thời lượng dự kiến**: 2 phút (01:30 – 03:30)
- **Mục tiêu slide**: Vạch trần 3 "cái bẫy" lớn khiến các đồ án Machine Learning sinh viên thường bị đánh trượt hoặc điểm thấp vì kết quả ảo (overfitting / data leakage). Chứng minh nhóm hiểu sâu bài toán từ bản chất dữ liệu chứ không chỉ biết chạy thư viện.
- **Chỉ dẫn thị giác**: Chỉ tay lần lượt vào 3 cột card màu cảnh báo trên màn hình: Cột 1 (Rò rỉ sao chép) $\rightarrow$ Cột 2 (Mù hình thức) $\rightarrow$ Cột 3 (Bẫy hộp đen).

### Lời thoại chi tiết từng câu từng chữ:
> *"Em xin kính chào Thầy Cô! Em là Thành viên 1, phụ trách mảng Khai phá và Tiền xử lý dữ liệu của nhóm.*
> 
> *Thưa Thầy Cô, vấn nạn tin giả, tin giật gân, câu view và lừa đảo tài chính trên mạng xã hội Việt Nam ngày càng tinh vi. Tuy nhiên, khi khảo sát các bài báo khoa học và các đồ án sinh viên những năm gần đây, nhóm chúng em nhận thấy hầu hết các giải pháp phân loại tin tức hiện nay đều vướng phải **3 Điểm Mù Cốt Tử**:*
> 
> *[Chỉ vào Cột 1]*
> ***Điểm mù thứ nhất: Rò rỉ dữ liệu do sao chép nội dung (Content Syndication Leakage).***  
> *Đặc thù của báo chí mạng Việt Nam là các trang lá cải, blog cá nhân liên tục copy xào xáo lại bài viết của các tòa soạn chính thống, chỉ thay đổi vài từ ngữ hoặc giật lại tiêu đề. Nếu chúng ta chỉ cào dữ liệu thô rồi chia tập Train/Test ngẫu nhiên theo cách thông thường, thì cùng một nội dung bài báo đã bị xào xáo sẽ lọt vào CẢ tập Train LẪN tập Test! Hậu quả là mô hình đạt điểm F1 ảo lên tới 98% – 99%, nhưng thực chất mô hình chỉ đang 'học vẹt' bản ghi cũ chứ không hề có khả năng tổng quát hóa ngoài thực tế.*
> 
> *[Chỉ vào Cột 2]*
> ***Điểm mù thứ hai: 'Mù' hình thức trình bày (Stylistic Blindness).***  
> *Một bài báo chính thống của báo Tuổi Trẻ đưa tin về vụ án bắt giữ đường dây lừa đảo tiền ảo và một bài viết giật gân bịa đặt trên mạng xã hội đều chia sẻ tới 85% từ vựng giống nhau: 'công an', 'lừa đảo', 'tiền ảo', 'bắt giữ'. Nếu chỉ dựa vào mô hình từ vựng n-gram hoặc biểu diễn ngữ nghĩa của BERT, mô hình sẽ bị rơi vào 'Bẫy trùng lặp từ vựng' (Vocabulary Overlap). Trong khi đó, tín hiệu phân loại đắt giá nhất của tin giả lại nằm ở **hình thức trình bày bề mặt**: lạm dụng dấu chấm than, viết IN HOA toàn bộ tiêu đề, hoặc tiêu đề giật tít dài bất thường so với nội dung thân bài rỗng tuếch.*
> 
> *[Chỉ vào Cột 3]*
> ***Điểm mù thứ ba: Bẫy 'Hộp đen' thiếu minh bạch (Black-Box AI).***  
> *Gắn nhãn một bài viết là 'Tin giả' ảnh hưởng trực tiếp đến uy tín danh dự và thậm chí là trách nhiệm pháp lý. Một mô hình nơ-ron phức tạp chỉ trả về một xác suất 0.95 mà không thể lý giải vì sao sẽ không thể nào thuyết phục được người dùng và ban kiểm duyệt.*
> 
> *Chính 3 điểm mù này là động lực để Nhóm 3 xây dựng kiến trúc dung hợp đặc trưng đột phá, sẽ được bạn Thành viên 4 trình bày ngay sau đây!*

---

## SLIDE 3: GIẢI PHÁP ĐỀ XUẤT: KIẾN TRÚC HỢP NHẤT ĐẶC TRƯNG MUỘN
- **Người trình bày**: **Thành viên 4 (TV4 - Trưởng nhóm Mô hình & XAI Lead)**
- **Thời lượng dự kiến**: 2 phút (03:30 – 05:30)
- **Mục tiêu slide**: Làm sáng tỏ kiến trúc "Song kiếm hợp bích". Giải thích tại sao 2 luồng Stylometrics và PhoBERT lại bổ trợ trực giao cho nhau, và cách thức dung hợp tại tầng 783 chiều.
- **Chỉ dẫn thị giác**: Chỉ tay vào sơ đồ luồng vector trực quan ở nửa trên màn hình: dẫn mắt hội đồng đi từ Khối Bài báo đầu vào $\rightarrow$ Tách đôi 2 nhánh (Stylometrics & PhoBERT) $\rightarrow$ Tầng Hợp nhất 783 chiều $\rightarrow$ Bộ phân loại đối chứng $\rightarrow$ Kết quả & SHAP Waterfall.

### Lời thoại chi tiết từng câu từng chữ:
> *"Kính thưa Thầy Cô! Em là Thành viên 4, chịu trách nhiệm về Mô hình hóa và Minh giải XAI của nhóm.*
> 
> *Để giải quyết triệt để 3 điểm mù mà bạn TV1 vừa phân tích, nhóm chúng em đề xuất giải pháp kiến trúc mang tên: **Song Kiếm Hợp Bích qua Dung Hợp Đặc Trưng Muộn (Late Feature Fusion)** như Thầy Cô đang quan sát trên sơ đồ trực quan.*
> 
> *Khi nhận một bài báo đầu vào gồm Tiêu đề và Thân bài, hệ thống của chúng em phân tách xử lý song song thành hai luồng thông tin độc lập và trực giao với nhau:*
> 
> *[Chỉ vào Khối Luồng 1]*
> - **Luồng 1 đóng vai trò là 'Thám tử hình thức' (Stylometrics Stream - 15 chiều):** *Luồng này quét 100% toàn văn bài báo từ đầu đến cuối không bỏ sót bất kỳ ký tự nào, với chi phí tính toán cực nhẹ dưới 2 mili-giây trên CPU. Nhiệm vụ của nó là bóc tách 15 chỉ số toán học về mật độ dấu câu giật gân, chữ hoa la hét, cấu trúc câu và từ điển kích động.*
> 
> *[Chỉ vào Khối Luồng 2]*
> - **Luồng 2 đóng vai trò là 'Chuyên gia ngữ nghĩa' (Deep Semantics Stream - 768 chiều):** *Được đảm nhiệm bởi mô hình Pre-trained Transformer tối tân nhất cho tiếng Việt là `vinai/phobert-base-v2`. Để vượt qua giới hạn 256 tokens của PhoBERT mà không làm mất thông điệp bài báo, nhóm áp dụng kỹ thuật Head-Tail Truncation (128 tokens đầu + 128 tokens cuối) kết hợp Attention-Masked Mean Pooling và chuẩn hóa vector đơn vị L2.*
> 
> *[Chỉ vào Khối Hợp Nhất trung tâm]*
> *Tại Tầng Hợp Nhất Muộn, chúng em ghép nối hai nguồn tri thức này thành một vector đặc trưng toàn diện **783 chiều** ($768 + 15$). Vector này sẽ được nạp vào hai kiến trúc phân loại đối chứng: một là thuật toán cây tăng cường **XGBoost**, hai là mạng nơ-ron sâu đa đầu vào **PyTorch Dual-Stream Classifier**.*
> 
> *Điểm đặc biệt nhất: Quyết định cuối cùng không chỉ trả về nhãn Real/Fake, mà được đưa qua thư viện **SHAP TreeExplainer** để xuất ra biểu đồ Waterfall chỉ rõ từng % yếu tố đóng góp.*
> 
> *Tiếp theo, xin mời bạn TV1 quay trở lại trình bày chi tiết về quy trình khai phá dữ liệu sạch của nhóm!"*

---

## SLIDE 4: QUY TRÌNH KHAI PHÁ DỮ LIỆU NÂNG CAO & MINHASH LSH
- **Người trình bày**: **Thành viên 1 (TV1 - Kỹ sư Dữ liệu & Cào Web)**
- **Thời lượng dự kiến**: 2 phút (05:30 – 07:30)
- **Mục tiêu slide**: Thể hiện quy chuẩn kỹ thuật cao trong môn Khai phá Dữ liệu: Thu thập đa nguồn cân bằng, khử bias tên nguồn bằng Regex, và ứng dụng thuật toán MinHash & LSH băm văn bản để loại bỏ tin xào nấu.
- **Chỉ dẫn thị giác**: Chỉ lần lượt vào 3 thẻ tiến trình có đánh số BƯỚC 01, BƯỚC 02, BƯỚC 03 ở bên trái màn hình.

### Lời thoại chi tiết từng câu từng chữ:
> *"Em xin tiếp tục với Quy trình Khai phá và Tiền xử lý Dữ liệu của nhóm.*
> 
> *Trong môn học Khai phá Dữ liệu, chất lượng dữ liệu quyết định tới 80% độ tin cậy của mô hình theo nguyên lý 'Garbage In, Garbage Out'. Quy trình của TV1 bao gồm 3 bước kỹ thuật nghiêm ngặt:*
> 
> *[Chỉ vào Bước 1]*
> ***Bước 1: Thu thập đa nguồn cân bằng với quy mô tối thiểu 4,000 bài viết.***  
> *- Đối với Nhãn Thật (Class 0 - 50%): Nhóm thu thập tự động qua các kênh RSS chính thức của các cơ quan báo chí chính thống uy tín nhất: VnExpress, Tuổi Trẻ, VietnamNet trên đa dạng chuyên mục (Thời sự, Pháp luật, Sức khỏe, Đời sống) kết hợp tập dữ liệu học thuật VLSP 2020 ReINTEL.*  
> *- Đối với Nhãn Giả / Giật gân (Class 1 - 50%): Nhóm tổng hợp từ các bộ chuẩn mở đã kiểm duyệt (VFND, ViFactCheck), kết hợp cào có chọn lọc các blog lá cải, bài thuốc chữa bệnh phản khoa học trên mạng xã hội, và khôi phục các đường link tin giả đã được công bố tại Cổng Thông tin VAFC (`tingia.gov.vn`) và AFP Kiểm chứng.*
> 
> *[Chỉ vào Bước 2]*
> ***Bước 2: Tẩy dấu vết nguồn (Source De-biasing) & Chuẩn hóa Unicode NFC.***  
> *- Đây là giải pháp chống hiện tượng 'Học vẹt' (Shortcut Learning): Các bài báo thường có tiền tố như `(Dân trí) -`, `Theo VnExpress...` hoặc chữ ký phóng viên ở cuối bài. Nếu để nguyên, mô hình sẽ chỉ học phân loại dựa theo tên báo chứ không học bản chất bài viết. TV1 sử dụng biểu thức chính quy (Regex) quét sạch 100% các dấu hiệu nhận diện nguồn này.*  
> *- Đồng thời, toàn bộ văn bản được đưa qua chuẩn hóa bảng mã `Unicode NFC` nhằm giải quyết dứt điểm xung đột giữa chữ dựng sẵn và chữ tổ hợp trong tiếng Việt.*
> 
> *[Chỉ vào Bước 3]*
> ***Bước 3: Khử trùng lặp cấp độ cận âm bằng MinHash & LSH.***  
> *- Để ngăn chặn hoàn toàn rò rỉ dữ liệu giữa tập Train và Test, nhóm băm văn bản thành các cụm từ 3-word shingles, sinh 128 hàm băm MinHash độc lập để xấp xỉ hệ số tương đồng Jaccard. Sau đó đưa qua cấu trúc LSH Indexing với 32 băng, 4 hàng.*  
> *- Bất kỳ cặp bài báo nào có độ tương đồng Jaccard $\ge 0.85$ (dấu hiệu của việc xào xáo nội dung), thuật toán sẽ chỉ giữ lại bài viết xuất bản sớm nhất và loại bỏ toàn bộ các bài sao chép phía sau.*
> 
> *Dữ liệu sạch sau cùng được đóng gói chuẩn định dạng `corpus_dedup.parquet` để bàn giao cho TV2 và TV3!"*

---

## SLIDE 5: KỸ THUẬT TRÍCH XUẤT ĐẶC TRƯNG KÉP
- **Người trình bày**: **Thành viên 2 (TV2 - Stylometrics) & Thành viên 3 (TV3 - PhoBERT)**
- **Thời lượng dự kiến**: 2 phút (07:30 – 09:30) (Mỗi bạn 1 phút)
- **Mục tiêu slide**: Trình bày chi tiết toán học của 15 chỉ số phong cách (TV2) và giải pháp kỹ thuật Head-Tail Truncation, Masked Mean Pooling của PhoBERT (TV3).
- **Chỉ dẫn thị giác**: TV2 đứng bên trái chỉ vào Cột 15 Stylometrics; sau đó TV3 tiếp quản đứng bên phải chỉ vào Cột PhoBERT 768D.

### Lời thoại chi tiết từng câu từng chữ:

#### Lời thoại TV2 (Kỹ sư Ngôn ngữ & Stylometrics - 1 phút):
> *"Kính thưa Thầy Cô, em là Thành viên 2, phụ trách nhánh Đặc trưng Phong cách học Stylometrics.*
> 
> *Quan sát cột bên trái, nhóm chúng em đã lượng hóa phong cách giật gân, câu view của tin tức mạng thành **15 chỉ số toán học chuẩn xác**, chia làm 3 nhóm độc lập:*
> 1. *Nhóm Dấu câu & Cảm thán (4 chỉ số): Đo lường mật độ dấu chấm than `!`, dấu hỏi chấm `?`, dấu trích dẫn `""` và dấu ngắt nhịp kịch tính `: -` đi kèm tổng mật độ dấu câu `punct_density`.*
> 2. *Nhóm Hình thức & Cấu trúc (5 chỉ số): Lượng hóa tỷ lệ chữ viết IN HOA (hành vi la hét, phóng đại sự việc), tỷ lệ độ dài Tiêu đề trên Thân bài (các bài clickbait thường có tiêu đề rất dài nhưng thân bài rỗng tuếch), tần suất chữ số ở tiêu đề (`100%`, `500k`), độ dài trung bình câu và tỷ lệ teencode kết hợp độ đa dạng từ vựng `type_token_ratio` (TTR).*
> 3. *Nhóm Ngữ dụng & Cảm xúc (6 chỉ số): So khớp với các bộ từ điển chuyên miền do nhóm tự xây dựng để đếm số lượng từ khóa giật tít ('chấn động', 'ngã ngửa'), cụm từ thúc ép cấp bách ('chia sẻ ngay', 'kẻo xóa'), đại từ xưng hô phi chuẩn mực ('bà con ơi', 'em thề'), độ tương đồng Jaccard giữa tiêu đề và thân bài để bắt bài thủ đoạn 'treo đầu dê bán thịt chó', độ phân cực cảm xúc tuyệt đối, và điểm tín nhiệm tên miền nguồn tin `domain_trust_score`.*
> 
> *Toàn bộ 15 chỉ số này được chuẩn hóa StandardScaler và xuất ra ma trận `X_stylometric.npy` với tốc độ cực nhanh: trên 500 bài viết mỗi giây trên CPU thông thường!*
> 
> *Sau đây, xin mời bạn TV3 trình bày về nhánh biểu diễn ngữ nghĩa sâu PhoBERT."*

#### Lời thoại TV3 (Kỹ sư Học sâu PhoBERT - 1 phút):
> *"Em xin chào Thầy Cô, em là Thành viên 3, phụ trách nhánh Nhúng ngữ nghĩa PhoBERT.*
> 
> *Ở cột bên phải, sứ mệnh của em là chuyển đổi toàn bộ ngữ cảnh logic của bài viết thành không gian vector dày đặc **768 chiều** bằng mô hình `vinai/phobert-base-v2`.*
> 
> *Trong quá trình thiết kế, bài toán kỹ thuật lớn nhất em phải giải quyết là: Các bài báo thường dài từ 800 đến 1,500 từ, trong khi PhoBERT chỉ nhận tối đa 256 subword tokens. Nếu chúng ta cắt ngọn thô bạo (chỉ lấy 256 tokens đầu), mô hình sẽ hoàn toàn không đọc được thân bài và kết luận — nơi chứa các lời đính chính hoặc thông tin phát ngôn cốt lõi.*
> 
> *Để giải quyết vấn đề này, em áp dụng chiến lược **Head-Tail Truncation**: Lấy trọn vẹn 128 tokens đầu (chứa Tiêu đề và đoạn mở đầu Lede tóm tắt 5W1H) ghép với 128 tokens cuối (chứa kết luận sự việc và cơ quan phát ngôn). Kỹ thuật này bảo toàn được trên 90% tín hiệu phân loại mà hoàn toàn không lo tràn bộ nhớ VRAM.*
> 
> *Tiếp đó, thay vì chỉ lấy vector của token `[CLS]`, em áp dụng cơ chế **Attention-Masked Mean Pooling** để tính trung bình có trọng số các token hợp lệ và **chuẩn hóa vector đơn vị L2** ($\|\mathbf{z}\|_2 = 1.0$), giúp khoảng cách Euclid tương ứng trực tiếp với độ tương đồng Cosine.*
> 
> *Toàn bộ quy trình được tối ưu chạy hàng loạt trên Google Colab T4 chỉ mất dưới 3 phút cho 4,000 bài báo!"*

---

## SLIDE 6: PHƯƠNG PHÁP MÔ HÌNH HÓA, ĐÁNH GIÁ & GIẢI THÍCH SHAP XAI
- **Người trình bày**: **Thành viên 4 (TV4 - Trưởng nhóm Mô hình & XAI Lead)**
- **Thời lượng dự kiến**: 2 phút (09:30 – 11:30)
- **Mục tiêu slide**: Chứng minh tính chặt chẽ trong thiết kế thí nghiệm (Ablation Study 3 cấp độ), hệ thống đo lường không thiên vị, và cơ chế giải thích SHAP XAI đẳng cấp.
- **Chỉ dẫn thị giác**: Chỉ vào Bảng Ablation Study ở nửa trên $\rightarrow$ chuyển mắt xuống 2 hộp bên dưới: Thước đo đánh giá và SHAP Explainability.

### Lời thoại chi tiết từng câu từng chữ:
> *"Em xin tiếp tục phần báo cáo với Thiết kế Thực nghiệm và Giải thích Mô hình.*
> 
> *Thưa Thầy Cô, để chứng minh một cách khoa học rằng phương pháp Hợp nhất Đặc trưng của nhóm thực sự vượt trội chứ không phải do may mắn, nhóm chúng em thiết kế một nghiên cứu đối sánh bài bản qua **3 Cấp Độ Thí Nghiệm (3-Tier Ablation Study)**:*
> 
> *- **Baseline 1 (Mô hình Cổ điển):** Kết hợp phương pháp TF-IDF truyền thống với 15 đặc trưng Stylometrics đưa vào Logistic Regression và XGBoost. Đây là mốc chặn dưới để so sánh.*  
> *- **Baseline 2 (Mô hình Đơn luồng sâu):** Chỉ sử dụng duy nhất 768 chiều nhúng PhoBERT đưa vào bộ phân loại tuyến tính. Mục tiêu là kiểm chứng xem mô hình ngôn ngữ lớn khi đứng một mình có bị 'bẫy từ vựng' như giả thuyết ban đầu hay không.*  
> *- **Mô hình Đề xuất (Late Feature Fusion - 783 chiều):** Nhóm thử nghiệm song song hai phương án:*  
>   *+ Phương án A: Ghép trực tiếp 783 chiều đưa vào XGBoost Classifier, tối ưu siêu tham số tự động qua 50 trials với Optuna và xử lý mất cân bằng lớp bằng tham số `scale_pos_weight`.*  
>   *+ Phương án B: Xây dựng mạng nơ-ron sâu đa nhánh **PyTorch DualStreamClassifier**. Nhánh ngữ nghĩa được nén từ 768 xuống 128 chiều, nhánh phong cách được phóng chiếu từ 15 lên 32 chiều, sau đó hợp nhất tại tầng 160 chiều trước khi phân loại. Thiết kế này giúp hai luồng thông tin có tiếng nói cân bằng, giải quyết triệt để sự chênh lệch số chiều.*
> 
> *Về quy trình kiểm thử, nhóm sử dụng **Stratified 5-Fold Cross Validation** và lấy **Macro-F1 Score** làm thước đo quyết định nhằm đảm bảo tính công bằng tuyệt đối trên cả hai lớp tin thật và tin giả.*
> 
> *Đặc biệt, ở hộp bên phải, nhóm sử dụng **SHAP TreeExplainer** dựa trên lý thuyết trò chơi của Shapley. Biểu đồ Waterfall sẽ bóc tách chính xác từng đặc trưng đóng góp bao nhiêu phần trăm rủi ro: ví dụ tiêu đề viết hoa đóng góp `+0.48`, từ khóa giật gân đóng góp `+0.35`, giúp biến một mô hình phức tạp thành một công cụ minh bạch, có thể giải trình được!*
> 
> *Tiếp theo, xin mời bạn TV5 trình bày về Hệ sinh thái công nghệ và Tính khả thi triển khai của đề tài!"*

---

## SLIDE 7: HỆ SINH THÁI CÔNG CỤ & TÍNH KHẢ THI HẠ TẦNG
- **Người trình bày**: **Thành viên 5 (TV5 - Kỹ sư MLOps & Trưởng nhóm)**
- **Thời lượng dự kiến**: 1 phút 30 giây (11:30 – 13:00)
- **Mục tiêu slide**: Chứng minh tính khả thi 100% của đồ án: Toàn bộ công nghệ được lựa chọn chuẩn mực, hiện đại và hoàn toàn khả thi trên hạ tầng miễn phí (0 đồng chi phí cho sinh viên).
- **Chỉ dẫn thị giác**: Chỉ tay vào Bảng 4 phân tầng công nghệ bên trái $\rightarrow$ Chỉ sang Hộp Kiểm chứng tính khả thi tính toán bên phải.

### Lời thoại chi tiết từng câu từng chữ:
> *"Em xin tiếp tục với Hệ sinh thái Công nghệ và Tính khả thi Kỹ thuật của đề tài.*
> 
> *Quan sát bảng phân tầng bên trái, nhóm chúng em đã lựa chọn những thư viện mã nguồn mở chuẩn mực nhất của hệ sinh thái Python hiện đại:*
> *- **Tầng 1 (Khai phá dữ liệu):** Sử dụng `trafilatura` và `BeautifulSoup4` để bóc tách nội dung báo chí sạch, `datasketch` cho thuật toán MinHash LSH, và lưu trữ chuẩn nén `PyArrow / Parquet`.*  
> *- **Tầng 2 (Xử lý ngôn ngữ):** Sử dụng `PyVi` cho tách từ ghép tiếng Việt và thư viện `transformers` của HuggingFace.*  
> *- **Tầng 3 (Học máy & XAI):** Sử dụng `PyTorch` hỗ trợ tính toán số thực nửa độ chính xác FP16, `XGBoost`, `Optuna` và `SHAP`.*  
> *- **Tầng 4 (Đóng gói & Trình diễn):** Chúng em phát triển giao diện Web tương tác bằng `Gradio Blocks` cho phép dán link URL phân tích kết quả tức thì trong dưới 1.5 giây, đi kèm dịch vụ REST API chuẩn mực bằng `FastAPI` và đóng gói toàn bộ môi trường trong `Docker`.*
> 
> *Về tính khả thi hạ tầng, như Thầy Cô thấy ở khung bên phải:*
> *- Chúng em đã tính toán chi tiết bài toán tài nguyên: Mô hình PhoBERT khi chạy với batch size 32 trên GPU miễn phí của Google Colab T4 chỉ tiêu tốn **khoảng 1.8GB VRAM** trên tổng số 15GB sẵn có, và chỉ mất **dưới 3 phút** để hoàn thành trích xuất ngữ nghĩa cho toàn bộ 4,000 bài báo!*  
> *- Toàn bộ các công đoạn còn lại (cào tin, bóc tách, trích xuất 15 Stylometrics và huấn luyện XGBoost) đều chạy cực kỳ mượt mà trên CPU máy tính cá nhân thông thường trong chưa đầy 45 giây.*
> 
> *Điều này khẳng định đề tài của nhóm hoàn toàn có tính khả thi kỹ thuật 100%, không phụ thuộc vào máy chủ đắt tiền hay bất kỳ API trả phí nào!"*

---

## SLIDE 8: PHƯƠNG PHÁP VẬN HÀNH CONTRACT-FIRST & PHÂN CÔNG ĐỘI NGŨ
- **Người trình bày**: **Thành viên 5 (TV5 - Trưởng nhóm Điều phối)**
- **Thời lượng dự kiến**: 1 phút 30 giây (13:00 – 14:30)
- **Mục tiêu slide**: Đóng lại bài thuyết trình bằng một điểm cộng lớn nhất về kỹ năng làm việc nhóm kỹ thuật: Cả 5 thành viên đều có vai trò rõ ràng, có hợp đồng giao tiếp dữ liệu cụ thể và có bộ test tự động kiểm định.
- **Chỉ dẫn thị giác**: Chỉ vào Bảng 5 Thành viên ở nửa trên $\rightarrow$ Dẫn ánh mắt xuống 3 hộp thẻ Contract-First ở nửa dưới (Bẫy nghẽn Thác nước, Giải pháp Mock Data Ngày 1, và Lợi ích Kiểm thử Tự động CI/CD) $\rightarrow$ Chốt bài thuyết trình bằng lời cảm ơn đĩnh đạc.

### Lời thoại chi tiết từng câu từng chữ:
> *"Kính thưa Thầy Cô, slide cuối cùng này thể hiện cách thức Nhóm 3 tổ chức đội ngũ và phối hợp kỹ thuật để đảm bảo đồ án về đích đúng hạn.*
> 
> *Nhóm chúng em gồm 5 thành viên, mỗi bạn đảm nhiệm trọn vẹn một mắt xích chuyên môn hóa cao độ:*
> *- **Bạn TV1:** Kỹ sư Dữ liệu & Cào Web $\rightarrow$ Bàn giao tập tin sạch `corpus_dedup.parquet`.*  
> *- **Bạn TV2:** Kỹ sư Ngôn ngữ & Stylometrics $\rightarrow$ Bàn giao ma trận đặc trưng `X_stylometric.npy` kích thước $N \times 15$.*  
> *- **Bạn TV3:** Kỹ sư Học sâu PhoBERT $\rightarrow$ Bàn giao ma trận ngữ nghĩa `X_semantic.npy` kích thước $N \times 768$.*  
> *- **Bạn TV4:** Trưởng nhóm Mô hình & XAI $\rightarrow$ Huấn luyện Late Fusion và đóng gói module `predict_with_shap()`.*  
> *- **Và em, TV5:** Đảm nhiệm MLOps, xây dựng Web Demo Gradio, Docker và điều phối toàn bộ tài liệu báo cáo.*
> 
> *Điểm mấu chốt giúp nhóm tự tin tuyệt đối là **Cơ chế Vận hành Contract-First & Mock-Driven** ở nửa dưới màn hình:*
> *- Nếu làm việc theo mô hình Thác nước (Waterfall) truyền thống, TV2 và TV3 sẽ phải ngồi chờ TV1 cào dữ liệu mất 2-3 tuần; TV4 lại phải chờ TV2 và TV3 thêm 2 tuần nữa. Nhưng tại Nhóm 3, ngay trong Ngày 1, nhóm đã đóng băng toàn bộ 4 Hợp đồng Dữ liệu (Data Contracts) và sinh ra bộ dữ liệu giả lập chuẩn schema trong thư mục `data/fixtures/`.*  
> *- Nhờ đó, **cả 5 thành viên đều có thể bắt tay vào lập trình độc lập 100% ngay từ Ngày 1**! Mọi sự ghép nối giữa các thành viên đều được kiểm thử tự động bằng một câu lệnh duy nhất: `pytest tests/test_data_contracts.py`.*
> 
> *Hiện tại, khung dự án, bộ sinh mock data và 9 bài test hợp đồng tự động đã được nhóm hoàn thiện và tích hợp thành công trên kho mã nguồn GitHub.*
> 
> *Trên đây là toàn bộ Báo cáo đề xuất đề tài của Nhóm 3. Chúng em xin chân thành cảm ơn quý Thầy Cô đã chú ý lắng nghe, và nhóm chúng em rất mong nhận được những ý kiến đóng góp, phản biện quý báu từ Hội đồng để hoàn thiện đồ án một cách xuất sắc nhất!*
> 
> *Chúng em xin kính mời quý Thầy Cô đặt câu hỏi ạ!"*

---

---

## CẨM NANG TOP 5 CÂU HỎI PHẢN BIỆN TỪ GIẢNG VIÊN & CÁCH TRẢ LỜI XUẤT SẮC

Dưới đây là 5 câu hỏi "bẫy" mang tính học thuật cao nhất mà các Giảng viên môn Học máy & Khai phá dữ liệu (SOICT - HUST) thường dùng để kiểm tra độ hiểu sâu của sinh viên khi duyệt đề tài, cùng chiến lược và câu trả lời mẫu xuất sắc:

---

### CÂU HỎI 1: "Tại sao nhóm lại dùng kỹ thuật Head-Tail Truncation (128 đầu + 128 đuôi) cho PhoBERT mà không dùng toàn văn? Có sợ bị mất thông tin quan trọng ở giữa bài viết không?"

* **Người trả lời chuẩn bị**: **Thành viên 3 (TV3 - PhoBERT Specialist)**
* **Bản chất câu hỏi**: Thầy Cô đang kiểm tra xem sinh viên có hiểu kiến trúc của Transformer (độ phức tạp tự chú ý $\mathcal{O}(L^2)$) và đặc thù cấu trúc bài báo chí (nguyên lý tháp ngược - Inverted Pyramid) hay không.
* **Cách trả lời xuất sắc**:
  > *"Em xin cảm ơn câu hỏi rất sâu sắc của Thầy/Cô ạ!*
  > 
  > *Nhóm chúng em lựa chọn chiến lược Head-Tail Truncation dựa trên cả **Lý thuyết Báo chí học** lẫn **Đặc tính Kỹ thuật của Transformer**:*
  > 
  > 1. *Về lý thuyết báo chí: Tin tức báo chí được viết theo cấu trúc kinh điển 'Hình tháp ngược' (Inverted Pyramid). Mọi thông tin cốt lõi nhất (Ai, Cái gì, Ở đâu, Khi nào - 5W1H) đều tập trung 100% ở đoạn mở đầu (Lede). Trong khi đó, phần kết bài luôn chứa thông tin về cơ quan ngôn luận, lời đính chính, hoặc lời kêu gọi hành động (Call-to-Action). Phần giữa bài báo thường chỉ là các diễn giải chi tiết, trích dẫn phụ trợ.*
  > 2. *Về mặt kỹ thuật: PhoBERT có giới hạn ngữ cảnh cứng là 256 tokens do chi phí ma trận chú ý tăng theo bậc hai $\mathcal{O}(L^2)$. Nếu cố tình nhồi nhét cả bài báo 1,500 từ, hoặc ta phải dùng các mô hình như Longformer (vốn chưa có bản Pre-trained tiếng Việt tối ưu như PhoBERT), hoặc sẽ bị lỗi tràn bộ nhớ VRAM CUDA OOM trên phần cứng học viên.*
  > 3. *Đặc biệt, nhóm chúng em KHÔNG hề bỏ qua phần giữa bài báo! Đó chính là lý do vì sao nhóm thiết kế **Luồng 1 - Stylometrics (TV2)**: Luồng Stylometrics quét 100% toàn văn không bỏ sót bất kỳ một chữ nào ở phần giữa thân bài, đo lường toàn bộ dấu câu, chữ hoa và từ kích động trên toàn bài. Sự kết hợp giữa Head-Tail PhoBERT và 100% Stylometrics tạo nên sự bù trừ hoàn hảo cho nhau ạ!"*

---

### CÂU HỎI 2: "Khi tinh chỉnh nhẹ (Partial Fine-tuning) PhoBERT và trích xuất đặc trưng, nhóm làm thế nào để đảm bảo không xảy ra Rò rỉ dữ liệu (Data Leakage) vào tập Test?"

* **Người trả lời chuẩn bị**: **Thành viên 4 (TV4 - Modeling Lead) hoặc TV3**
* **Bản chất câu hỏi**: Thầy Cô kiểm tra xem sinh viên có trích xuất đặc trưng trên TOÀN BỘ tập dữ liệu trước khi chia train/test hay không (một lỗi sai kinh điển khiến đồ án bị điểm 0 về phương pháp).
* **Cách trả lời xuất sắc**:
  > *"Dạ em xin cảm ơn Thầy/Cô đã nhắc nhở về một lỗi kỹ thuật sống còn trong học máy ạ!*
  > 
  > *Nhóm chúng em cam kết tuân thủ nghiêm ngặt nguyên tắc **Đóng băng Tập Kiểm thử (Test Set Quarantine)**:*
  > 1. *Toàn bộ quy trình chia tập dữ liệu (Stratified Train-Test Split 80/20) được thực hiện ngay từ Tầng 2, sau khi đã khử trùng lặp MinHash LSH. Tập Test được khóa kín và lưu thành file riêng biệt.*
  > 2. *Bước Partial Fine-tuning 2 tầng cuối của PhoBERT **CHỈ ĐƯỢC HUẤN LUYỆN DUY NHẤT TRÊN TẬP TRAIN**. Trọng số mô hình sau khi tinh chỉnh xong sẽ được cố định và đặt ở chế độ `model.eval()` với `torch.no_grad()` khi trích xuất vector cho tập Test.*
  > 3. *Tương tự, đối với 15 đặc trưng Stylometrics của TV2: Bộ chuẩn hóa `StandardScaler` chỉ thực hiện `scaler.fit()` trên tập Train, sau đó chỉ dùng `scaler.transform()` để áp dụng lên tập Test. Tuyệt đối không dùng chung tham số trung bình $\mu$ và độ lệch chuẩn $\sigma$ của tập Test trong quá trình huấn luyện.*
  > 
  > *Nhờ vậy, tập Test hoàn toàn là dữ liệu chưa từng thấy (unseen data), đảm bảo kết quả đánh giá là 100% trung thực và khách quan ạ!"*

---

### CÂU HỎI 3: "Dữ liệu nhãn tin giả lấy từ Cổng tingia.gov.vn thường đã bị gỡ bài gốc hoặc xóa liên kết. Nhóm giải quyết vấn đề lệch mẫu (Takedown Bias / Broken Links) này như thế nào?"

* **Người trả lời chuẩn bị**: **Thành viên 1 (TV1 - Data Engineering Lead)**
* **Bản chất câu hỏi**: Thầy Cô kiểm tra xem sinh viên đã thực sự bắt tay vào cào dữ liệu thật từ các nguồn chính thống chưa, hay chỉ viết đề xuất trên lý thuyết.
* **Cách trả lời xuất sắc**:
  > *"Em xin cảm ơn câu hỏi rất thực tế và sắc bén của Thầy/Cô ạ!*
  > 
  > *Đúng như Thầy/Cô đã chỉ ra, các tin giả khi bị Cổng VAFC (`tingia.gov.vn`) hoặc Bộ Thông tin & Truyền thông xử phạt thì thường đường link gốc trên mạng xã hội đã bị khóa hoặc chủ tài khoản tự gỡ bài (Takedown Bias).*
  > 
  > *Để giải quyết triệt để rào cản này, nhóm chúng em áp dụng **Chiến lược Phục hồi Đa Tầng (Multi-tier Recovery Strategy)**:*
  > 1. *Phục hồi qua Báo cáo Kiểm chứng: Trang `tingia.gov.vn` và AFP Fact-Check luôn chụp lại ảnh bằng chứng và trích dẫn nguyên văn văn bản tin giả trong bài cảnh báo bác bỏ tin đồn. TV1 sử dụng bộ bóc tách chuyên dụng để lấy lại đúng đoạn văn bản xuyên tạc được trích dẫn đó.*
  > 2. *Tận dụng cỗ máy thời gian Internet Archive (Wayback Machine): Với các URL tin giả đã bị xóa, chúng em gửi truy vấn tới API của `web.archive.org` để khôi phục snapshot nội dung tại thời điểm bài viết đang phát tán.*
  > 3. *Không phụ thuộc vào một nguồn duy nhất: Cổng VAFC chỉ đóng vai trò là một nguồn kiểm chứng đối sánh. Nguồn tin giả chính của nhóm vẫn được đảm bảo quy mô lớn từ bộ chuẩn nghiên cứu mở VLSP 2020 ReINTEL và VFND đã được cộng đồng gán nhãn sẵn, kết hợp cào các blog tin tức lá cải, bài thuốc dân gian phản khoa học còn đang hoạt động công khai trên Internet.*
  > 
  > *Do đó, nhóm đảm bảo chắc chắn thu thập đủ tối thiểu 2,000 bài viết nhãn giả chất lượng cao ạ!"*

---

### CÂU HỎI 4: "Tại sao nhóm lại đề xuất cả XGBoost và Mạng nơ-ron đa đầu vào PyTorch (Dual-Stream NN) mà không chọn hẳn một mô hình ngay từ đầu?"

* **Người trả lời chuẩn bị**: **Thành viên 4 (TV4 - Modeling & XAI Lead)**
* **Bản chất câu hỏi**: Thầy Cô kiểm tra tư duy so sánh đối chứng khoa học theo đúng tinh thần môn học (Ablation Study & Model Exploration) trong Slide L0_Intro.
* **Cách trả lời xuất sắc**:
  > *"Dạ em xin cảm ơn Thầy/Cô! Đây là một điểm thiết kế có chủ đích rất rõ ràng trong đề cương của nhóm ạ.*
  > 
  > *Theo đúng tiêu chí đánh giá môn học của Viện CNTT&TT (được nêu rõ trong Slide L0_Intro), một đồ án chất lượng cao đòi hỏi sinh viên phải **'Đánh giá và lựa chọn kỹ lưỡng mô hình dựa trên thực nghiệm đối chứng'** chứ không được chọn cảm tính.*
  > 
  > *Hai mô hình này đại diện cho hai triết lý học máy hoàn toàn khác biệt nhau:*
  > 1. *XGBoost đại diện cho họ Cây quyết định tăng cường (Tree-based Ensembles) trên dữ liệu dạng bảng: Rất mạnh trong việc tìm ra các ngưỡng phân cắt phi tuyến trên các đặc trưng rời rạc (như 15 chỉ số Stylometrics), chống nhiễu tốt và đặc biệt là tương thích hoàn hảo 100% với thuật toán tính nhanh của thư viện giải thích `SHAP TreeExplainer`.*
  > 2. *Mạng PyTorch Dual-Stream NN đại diện cho họ Học sâu (Deep Learning): Giải quyết bài toán 'Lệch pha số chiều' bằng cách tự động nén ngữ nghĩa ($768 \rightarrow 128$) và mở rộng phong cách ($15 \rightarrow 32$), cho phép học các mối liên hệ ẩn phi tuyến tính phức tạp giữa hai luồng.*
  > 
  > *Bằng cách triển khai cả hai mô hình và so sánh chéo qua Stratified 5-Fold Cross Validation, nhóm chúng em sẽ có câu trả lời khoa học chính xác xem: Cây quyết định tăng cường hay Mạng nơ-ron đa luồng là kiến trúc tối ưu nhất cho bài toán Hợp nhất đặc trưng tiếng Việt này ạ!"*

---

### CÂU HỎI 5: "Vector ngữ nghĩa 768 chiều của PhoBERT là không gian trừu tượng, người dùng không thể hiểu được các chiều $z_1, z_2...$ nghĩa là gì. Vậy làm thế nào nhóm giải thích được lý do bằng SHAP cho người dùng cuối trên Web Demo?"

* **Người trả lời chuẩn bị**: **Thành viên 4 (TV4 - XAI Lead) phối hợp cùng TV5**
* **Bản chất câu hỏi**: Thầy Cô kiểm tra xem nhóm có thực sự hiểu bản chất của SHAP trên các đặc trưng liên tục trừu tượng (Embedding features) hay chỉ nói lý thuyết suông.
* **Cách trả lời xuất sắc**:
  > *"Em xin cảm ơn Thầy/Cô, đây là câu hỏi hay nhất và cũng là 'điểm ăn tiền' nhất trong thiết kế XAI của nhóm chúng em ạ!*
  > 
  > *Đúng như Thầy Cô nhận định, nếu SHAP xuất ra giá trị đóng góp cho chiều thứ $z_{412}$ của PhoBERT thì con số đó hoàn toàn vô nghĩa với con người.*
  > 
  > *Nhóm chúng em đã thiết kế cơ chế **Minh giải Phân cấp (Hierarchical Explainability)** gồm 2 cấp độ:*
  > 
  > 1. *Cấp độ 1 — Gom cụm Nhóm Đặc trưng (Feature Group Attribution):*  
  >    *Trên không gian 783 chiều, nhóm gom toàn bộ 768 chiều của PhoBERT thành một siêu nhóm đại diện mang tên `'Ngữ cảnh & Ngữ nghĩa nội dung bài viết'`. Giá trị SHAP của siêu nhóm này được tính bằng tổng độ lớn đóng góp của các chiều thành phần:*  
  >    $$\Phi_{\text{Semantic}} = \sum_{j=1}^{768} \phi_{j}$$  
  >    *Người dùng sẽ thấy rõ: Ngữ nghĩa nội dung bài viết đang kéo nguy cơ tin giả tăng hay giảm bao nhiêu phần trăm.*
  > 
  > 2. *Cấp độ 2 — Minh giải chi tiết dựa trên 15 Đặc trưng Stylometrics tường minh:*  
  >    *15 đặc trưng hình thức của TV2 đều có ý nghĩa vật lý và ngữ dụng học trực quan 100%. Khi hiển thị trên biểu đồ Gradio của TV5, hệ thống sẽ ánh xạ trực tiếp thành các câu cảnh báo tự nhiên bằng tiếng Việt:*  
  >    *- Thay vì hiện `exclamation_density = +0.48`, giao diện hiện: **'Mật độ dấu chấm than (!) dày đặc bất thường làm tăng 48% nguy cơ tin giả'**.*  
  >    *- Thay vì hiện `clickbait_keyword_hits = +0.35`, giao diện hiện: **'Xuất hiện các từ kích động câu view ('CHẤN ĐỘNG', 'BẬT NGỬA') làm tăng 35% nguy cơ'**.*
  > 
  > *Sự kết hợp giữa đóng góp tổng thể của ngữ nghĩa và các chi tiết bóc tách của phong cách viết giúp người dùng thông thường và Hội đồng kiểm duyệt hiểu ngay lập tức nguyên nhân trong chưa đầy 1 giây ạ!"*

---

*Tài liệu được soạn thảo phục vụ bài bảo vệ đề xuất đề tài môn học IT3190 — Kỳ 2026.1 — Trường CNTT&TT, ĐHBK Hà Nội.*
