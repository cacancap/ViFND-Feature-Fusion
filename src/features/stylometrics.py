"""
ViFND - Vietnamese Fake News Detection
Module: src/features/stylometrics.py
Vai trò: Thành viên 2 (TV2) - Kỹ sư Ngôn ngữ & Đặc trưng Hình thức

Trích xuất 15 chỉ số hình thức (Stylometric & Linguistic Features) từ tiêu đề và nội dung bài báo:
1. upper_ratio
2. exclamation_density
3. question_density
4. punct_density
5. title_body_ratio
6. clickbait_keyword_hits
7. digit_in_title
8. colon_dash_density
9. quote_density
10. avg_sentence_len
11. urgency_density
12. pronoun_density
13. teencode_ratio
14. title_body_jaccard
15. sentiment_extremity
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import re
import json
import string
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "upper_ratio",
    "exclamation_density",
    "question_density",
    "punct_density",
    "title_body_ratio",
    "clickbait_keyword_hits",
    "digit_in_title",
    "colon_dash_density",
    "quote_density",
    "avg_sentence_len",
    "urgency_density",
    "pronoun_density",
    "teencode_ratio",
    "title_body_jaccard",
    "sentiment_extremity"
]

class StylometricExtractor:
    """Trích xuất 15 đặc trưng phong cách hình thức và ngôn ngữ học tiếng Việt."""

    def __init__(self, lexicons_dir: Optional[Union[str, Path]] = None):
        if lexicons_dir is None:
            current_dir = Path(__file__).resolve().parent
            self.lexicons_dir = current_dir / "lexicons"
        else:
            self.lexicons_dir = Path(lexicons_dir)

        self.clickbait_lexicon = self._load_lexicon("clickbait_keywords.txt")
        self.urgency_lexicon = self._load_lexicon("urgency_keywords.txt")
        self.pronoun_lexicon = self._load_lexicon("colloquial_pronouns.txt")
        self.teencode_lexicon = self._load_lexicon("teencode_dictionary.txt")
        
        # Heuristic sentiment sets
        self.pos_words = {
            "tuyệt_vời", "kỳ_diệu", "thần_thánh", "thành_công", "hạnh_phúc", "vinh_quang",
            "tự_hào", "thắng_lợi", "xuất_sắc", "đỉnh_cao", "phép_màu", "thần_kỳ", "tốt_đẹp",
            "yêu_thương", "rực_rỡ", "ấm_áp", "hoàn_hảo", "vui_sướng"
        }
        self.neg_words = {
            "kinh_hoàng", "thảm_họa", "chết_chóc", "đẫm_máu", "rùng_rợn", "khủng_khiếp",
            "dã_man", "man_rợ", "phẫn_nộ", "tang_thương", "chết_oan", "bị_lừa", "sụp_đổ",
            "oan_ức", "đau_đớn", "cay_đắng", "tuyệt_vọng", "thê_thảm", "tiêu_đời"
        }

        self.scaler = StandardScaler()
        self.is_fitted = False

    def _load_lexicon(self, filename: str) -> set:
        path = self.lexicons_dir / filename
        if not path.exists():
            return set()
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
        return set(lines)

    @staticmethod
    def tokenize_text(text: str) -> str:
        """Tách từ ghép tiếng Việt bằng PyVi ViTokenizer."""
        if not text or not isinstance(text, str):
            return ""
        return ViTokenizer.tokenize(text)

    def extract_single(self, title: str, content: str) -> np.ndarray:
        """
        Trích xuất 15 đặc trưng cho 1 cặp (title, content).
        Trả về vector numpy shape (15,) kiểu np.float32.
        """
        title = title if isinstance(title, str) else ""
        content = content if isinstance(content, str) else ""
        full_text = f"{title} {content}".strip()

        # Tiền xử lý đếm cơ bản
        num_chars_full = max(len(full_text), 1)
        num_chars_title = len(title)
        
        # Tokenized representation
        tokenized_title = self.tokenize_text(title)
        tokenized_content = self.tokenize_text(content)
        tokenized_full = f"{tokenized_title} {tokenized_content}".strip()

        words_title = tokenized_title.split() if tokenized_title else []
        words_content = tokenized_content.split() if tokenized_content else []
        words_full = tokenized_full.split() if tokenized_full else []

        n_words_title = len(words_title)
        n_words_content = len(words_content)
        n_words_full = len(words_full)

        # Đếm câu thô qua dấu chấm, chấm hỏi, chấm than
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
        n_sentences = max(len(sentences), 1)

        # 1. upper_ratio: Tỷ lệ ký tự viết hoa trong toàn bài
        upper_count = sum(1 for c in full_text if c.isupper())
        feat_1 = upper_count / num_chars_full

        # 2. exclamation_density: Mật độ dấu chấm than trên tổng số từ
        excl_count = full_text.count('!')
        feat_2 = excl_count / (n_words_full + 1)

        # 3. question_density: Mật độ dấu chấm hỏi trên tổng số từ
        ques_count = full_text.count('?')
        feat_3 = ques_count / (n_words_full + 1)

        # 4. punct_density: Mật độ dấu câu tổng thể
        punct_count = sum(1 for c in full_text if c in string.punctuation)
        feat_4 = punct_count / num_chars_full

        # 5. title_body_ratio: Tỷ số độ dài từ giữa tiêu đề và thân bài
        feat_5 = n_words_title / max(n_words_content, 1)

        # 6. clickbait_keyword_hits: Tổng số lần xuất hiện từ khóa giật gân
        lower_full = full_text.lower()
        cb_hits = sum(lower_full.count(kw) for kw in self.clickbait_lexicon)
        feat_6 = float(cb_hits)

        # 7. digit_in_title: Số chữ số xuất hiện trong tiêu đề
        digit_count = sum(1 for c in title if c.isdigit())
        feat_7 = float(digit_count)

        # 8. colon_dash_density: Mật độ dấu hai chấm và gạch ngang trên mỗi câu
        colon_dash = full_text.count(':') + full_text.count('-')
        feat_8 = colon_dash / (n_sentences + 1)

        # 9. quote_density: Mật độ dấu ngoặc kép trên số từ
        quote_count = full_text.count('"') + full_text.count('”') + full_text.count('“') + full_text.count("'")
        feat_9 = quote_count / (n_words_full + 1)

        # 10. avg_sentence_len: Độ dài trung bình câu theo số từ
        feat_10 = n_words_full / (n_sentences + 1)

        # 11. urgency_density: Số lần xuất hiện từ khóa cấp bách / kêu gọi chia sẻ
        urgency_hits = sum(lower_full.count(kw) for kw in self.urgency_lexicon)
        feat_11 = float(urgency_hits)

        # 12. pronoun_density: Số lần xuất hiện đại từ nhân xưng bình dân
        pronoun_hits = sum(lower_full.count(p) for p in self.pronoun_lexicon)
        feat_12 = float(pronoun_hits)

        # 13. teencode_ratio: Tỷ lệ từ lóng / viết tắt mạng xã hội
        teencode_hits = sum(1 for w in words_full if w.lower() in self.teencode_lexicon)
        feat_13 = teencode_hits / (n_words_full + 1)

        # 14. title_body_jaccard: Độ tương đồng từ vựng tiêu đề vs thân bài
        vocab_title = set(w.lower() for w in words_title)
        vocab_body = set(w.lower() for w in words_content)
        union_vocab = vocab_title.union(vocab_body)
        if union_vocab:
            feat_14 = len(vocab_title.intersection(vocab_body)) / len(union_vocab)
        else:
            feat_14 = 0.0

        # 15. sentiment_extremity: Độ lệch cảm xúc cực đoan |P_pos - P_neg|
        pos_hits = sum(tokenized_full.lower().count(pw) for pw in self.pos_words)
        neg_hits = sum(tokenized_full.lower().count(nw) for nw in self.neg_words)
        tot_sent = max(pos_hits + neg_hits, 1)
        feat_15 = abs((pos_hits - neg_hits) / tot_sent)

        vector = np.array([
            feat_1, feat_2, feat_3, feat_4, feat_5,
            feat_6, feat_7, feat_8, feat_9, feat_10,
            feat_11, feat_12, feat_13, feat_14, feat_15
        ], dtype=np.float32)

        # Bảo vệ tuyệt đối chống NaN / Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return vector

    def extract_batch(self, df: pd.DataFrame, title_col: str = "title", content_col: str = "content") -> np.ndarray:
        """
        Trích xuất 15 đặc trưng cho toàn bộ DataFrame bài báo.
        Trả về ma trận numpy shape (N, 15) dtype np.float32.
        """
        results = []
        for _, row in df.iterrows():
            t = str(row.get(title_col, ""))
            c = str(row.get(content_col, ""))
            vec = self.extract_single(t, c)
            results.append(vec)
        
        matrix = np.vstack(results).astype(np.float32)
        return matrix

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Huấn luyện StandardScaler và chuẩn hóa Z-score."""
        scaled = self.scaler.fit_transform(X).astype(np.float32)
        self.is_fitted = True
        return scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Áp dụng StandardScaler đã huấn luyện."""
        if not self.is_fitted:
            return X.astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return FEATURE_NAMES.copy()

    def save_feature_names(self, filepath: Union[str, Path]):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(FEATURE_NAMES, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    extractor = StylometricExtractor()
    sample_title = "CHẤN ĐỘNG: Lộ bài thuốc thần kỳ chữa dứt điểm ung thư 100% sau 3 ngày!!!"
    sample_content = "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ! Em thề thuốc tiên này cứu cả nhà đó, ai ko tin thì tùy nhé."
    
    vec = extractor.extract_single(sample_title, sample_content)
    print(f"Trích xuất thành công {len(vec)} đặc trưng:")
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  - {name:25s}: {val:.4f}")
