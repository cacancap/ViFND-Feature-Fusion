# ĐẶC TẢ VAI TRÒ & QUY TRÌNH KỸ THUẬT: THÀNH VIÊN 3 (TV3)
## KỸ SƯ HỌC SÂU & NHÚNG NGỮ NGHĨA PHOBERT (Deep Semantics & PhoBERT Specialist)

**Phạm vi phụ trách**: Tầng 4A (Biểu diễn ngữ nghĩa sâu qua Pre-trained Transformer).  
**Thư mục làm việc chính**: `src/features/`, Google Colab Notebooks  
**Tập tin bàn giao chính**: 
1. `data/processed/features/X_semantic.npy` $(N \times 768)$  
2. Module Python: `src/models/phobert_embedder.py`  

---

## 1. MỤC TIÊU VÀ SỨ MỆNH KHOA HỌC
Sứ mệnh của TV3 là chuyển đổi toàn bộ ngữ cảnh, ngữ pháp và nội dung bài báo tiếng Việt thành một **không gian vector ngữ nghĩa dày đặc 768 chiều (Dense Contextual Embedding)**:
- Sử dụng mô hình Transformer tối tân dành riêng cho tiếng Việt: `vinai/phobert-base-v2`.
- Giải quyết bài toán kỹ thuật hóc búa: **Giới hạn độ dài ngữ cảnh 256 tokens** của PhoBERT đối với các bài báo dài 1,000 từ mà không làm mất thông tin quan trọng hay gây lỗi tràn bộ nhớ VRAM (CUDA Out Of Memory).

---

## 2. QUY TRÌNH KỸ THUẬT CỦA TV3

### Bước 1: Khởi Tạo Mô Hình & Tokenizer PhoBERT
- Tải mô hình và tokenizer chính thức từ HuggingFace:
  ```python
  from transformers import AutoModel, AutoTokenizer
  MODEL_NAME = "vinai/phobert-base-v2"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModel.from_pretrained(MODEL_NAME)
  ```
- **Lưu ý đặc biệt**: PhoBERT yêu cầu đầu vào văn bản đã được tách từ ghép bằng dấu gạch dưới (`_`). Do đó, TV3 bắt buộc phải nhận đầu vào từ file `corpus_tokenized.parquet` do TV2 xử lý (hoặc dùng hàm tách từ của TV2).

### Bước 2: Chiến Lược Cắt Tỉa Head-Tail Truncation (Vượt Qua Rào Cản 256 Tokens)
Một bài báo trung bình có $600 - 1,500$ từ, trong khi PhoBERT chỉ nhận tối đa 256 subword tokens:
- **Nếu cắt ngọn thô bạo (`tokens[:256]`)**: Mô hình sẽ hoàn toàn không đọc được phần thân và phần kết bài viết (nơi chứa các lời đính chính, nguồn tin, hoặc kết luận sự việc).
- **Giải pháp kỹ thuật của TV3**: Áp dụng kỹ thuật **Head-Tail Truncation**:
  - Lấy 128 tokens đầu tiên (chứa Tiêu đề và Đoạn mở đầu - Lede: tóm tắt 5W1H của bài báo).
  - Lấy 128 tokens cuối cùng (chứa Đoạn kết, lời kêu gọi hành động, tên tác giả, cơ quan ngôn luận).
  $$\text{Input Sequence} = [\text{Token}_{1:128}] \cup [\text{Token}_{L-128:L}]$$
  Chiến lược này bảo toàn được 90%+ tín hiệu phân loại của toàn bộ bài viết trong giới hạn 256 tokens.

### Bước 3: Cơ Chế Attention-Masked Mean Pooling
Thay vì chỉ lấy vector của token `[CLS]` (dễ bị lệch khi không fine-tune toàn bộ mạng), TV3 sử dụng cơ chế **Mean Pooling có mặt nạ chú ý**:
$$\mathbf{z}_{\text{sem}} = \frac{\sum_{i=1}^L m_i \cdot \mathbf{h}_i}{\sum_{i=1}^L m_i}$$
trong đó:
- $\mathbf{h}_i \in \mathbb{R}^{768}$ là vector biểu diễn ẩn ở tầng cuối cùng (layer 12) của token thứ $i$.
- $m_i \in \{0, 1\}$ là giá trị attention mask (1 đối với token thật, 0 đối với padding).

### Bước 4: Chuẩn Hóa Vector Đơn Vị L2 (L2 Normalization)
Để khoảng cách giữa các bài viết trong không gian 768 chiều tương ứng trực tiếp với độ tương đồng Cosine (Cosine Similarity), mỗi vector bài viết bắt buộc phải được chuẩn hóa về độ dài bằng 1:
$$\mathbf{z}_{\text{norm}} = \frac{\mathbf{z}_{\text{sem}}}{\|\mathbf{z}_{\text{sem}}\|_2 + 10^{-8}}$$

### Bước 5: Thực Thi Hàng Loạt Trên GPU Colab T4 & Đóng Gói Module
- Viết notebook chạy batch inference trên Google Colab T4 (batch size = 32, bật `torch.no_grad()`).
- Xuất toàn bộ ma trận ra file nhị phân: `data/processed/features/X_semantic.npy`.
- Đóng gói class suy luận phục vụ dự đoán thời gian thực cho TV5:
  ```python
  class PhoBertEmbedder:
      def encode_single_article(self, title: str, content: str) -> np.ndarray:
          """Trả về vector numpy shape (768,) chuẩn L2 trong thời gian < 150ms."""
          ...
  ```

---

## 3. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE - DOD)
- [ ] Vượt qua bài kiểm tra tự động: `pytest tests/test_data_contracts.py::test_contract_p3`.
- [ ] Ma trận `X_semantic.npy` có kích thước chính xác $(N, 768)$, kiểu dữ liệu `np.float32`.
- [ ] 100% các hàng đều thỏa mãn chuẩn L2: $\|\mathbf{z}\|_2 = 1.0 \pm 10^{-4}$.
- [ ] Thời gian mã hóa 4,000 bài báo mất dưới **3 phút trên GPU Colab T4**.
- [ ] Hàm `encode_single_article` chạy độc lập trên CPU mất dưới **250ms**.
