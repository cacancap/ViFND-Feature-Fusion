"""
ViFND - Vietnamese Fake News Detection
Module: src/features/cross_verifier.py
Tầng: Cross-Source Corroboration Engine (Tầng Kiểm chứng Chéo Đa Nguồn - Hybrid)

Chức năng:
1. Kiểm chứng chéo sự kiện với Kho báo chí chính thống cục bộ (Local Index: BM25/TF-IDF trên tập nhãn 0).
2. Fallback tìm kiếm đối chứng thời gian thực (Google News RSS / Web News) khi cần hoặc khi demo online.
3. Trích xuất 4 đặc trưng kiểm chứng toán học cho Late Fusion:
   - veri_max_corroboration_sim: Độ tương đồng cao nhất với bài báo chính thống (0.0 - 1.0)
   - veri_verified_source_count: Số lượng nguồn báo chí lớn cùng đưa tin (chuẩn hóa 0.0 - 1.0)
   - veri_headline_cross_stance: Độ nhất quán giữa tiêu đề truy vấn và tiêu đề chính thống (0.0 - 1.0)
   - veri_unverified_anomaly_penalty: Điểm phạt khi tin có dấu hiệu giật gân nhưng 0 nguồn chính thống nào xác nhận
4. Trả về danh sách bài báo đối chứng (Evidence Articles) để hiển thị trực quan lên Web UI.
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
import xml.etree.ElementTree as ET
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

CROSS_VERIFICATION_FEATURE_NAMES = [
    "veri_max_corroboration_sim",
    "veri_verified_source_count",
    "veri_headline_cross_stance",
    "veri_unverified_anomaly_penalty"
]

VERI_FEATURE_DESCRIPTIONS = {
    "veri_max_corroboration_sim": "Mức độ tương đồng nội dung với các bài báo từ nguồn chính thống.",
    "veri_verified_source_count": "Số lượng cơ quan báo chí uy tín cùng đưa tin về sự kiện.",
    "veri_headline_cross_stance": "Độ nhất quán giữa tiêu đề truy vấn và thông tin từ báo chí chính thống.",
    "veri_unverified_anomaly_penalty": "Điểm phạt bất thường khi đưa tin giật gân nhưng không có nguồn chính thống nào xác nhận."
}

TRUSTED_DOMAINS = [
    "vnexpress.net", "tuoitre.vn", "vietnamnet.vn", "thanhnien.vn",
    "dantri.com.vn", "vtv.vn", "baochinhphu.vn", "nhandan.vn",
    "cafef.vn", "plo.vn", "sggp.org.vn", "chinhphu.vn", "tingia.gov.vn"
]


class CrossSourceVerifier:
    """
    Bộ kiểm chứng chéo đa nguồn Hybrid kết hợp kho cục bộ và tìm kiếm đối chứng.
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        enable_web_search: bool = True,
        web_search_timeout: float = 3.0
    ):
        self.enable_web_search = enable_web_search
        self.web_search_timeout = web_search_timeout
        self.corpus_path = corpus_path
        self.corpus_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_tfidf = None
        
        self._initialize_local_index()

    def _initialize_local_index(self):
        """Khởi tạo kho ngữ liệu chính thống và bộ lập chỉ mục TF-IDF."""
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate_paths = []
        if self.corpus_path:
            candidate_paths.append(Path(self.corpus_path))
        
        candidate_paths.extend([
            project_root / "data" / "processed" / "corpus_dedup.parquet",
            project_root / "data" / "fixtures" / "sample_corpus.parquet",
            project_root / "data" / "fixtures" / "sample_corpus.csv"
        ])

        target_file = None
        for p in candidate_paths:
            if p.exists():
                target_file = p
                break

        if target_file is not None:
            try:
                if str(target_file).endswith(".parquet"):
                    df = pd.read_parquet(target_file)
                else:
                    df = pd.read_csv(target_file)
                
                # Chỉ lập chỉ mục các bài báo chính thống (label == 0)
                if "label" in df.columns:
                    verified_df = df[df["label"] == 0].copy()
                else:
                    verified_df = df.copy()

                if len(verified_df) > 0:
                    self.corpus_df = verified_df.reset_index(drop=True)
                    # Tạo trường văn bản kết hợp để lập chỉ mục
                    combined_texts = (
                        self.corpus_df["title"].fillna("") + " " +
                        self.corpus_df["content"].fillna("")
                    ).tolist()

                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2)
                    )
                    self.corpus_tfidf = self.tfidf_vectorizer.fit_transform(combined_texts)
            except Exception as e:
                print(f"[CrossSourceVerifier] Lỗi khi lập chỉ mục kho cục bộ: {e}")

    def _clean_query_keywords(self, title: str) -> str:
        """Trích xuất từ khóa trọng tâm từ tiêu đề bài viết để truy vấn."""
        # Loại bỏ các tiền tố giật gân, dấu câu thừa
        clean = re.sub(r'^(CHẤN ĐỘNG|CẢNH BÁO|BÍ MẬT|BẬT MÍ|TIN NÓNG|NÓNG|HOT|HÉ LỘ)[\s\:\!\?]+', '', title, flags=re.IGNORECASE)
        clean = re.sub(r'[!\?\"\'\(\)\[\]\{\}\#\$\%]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        words = clean.split()
        if len(words) > 12:
            clean = " ".join(words[:12])
        return clean

    def _query_local_corpus(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Truy vấn kho lưu trữ báo chính thống cục bộ."""
        if self.corpus_df is None or self.tfidf_vectorizer is None or self.corpus_tfidf is None:
            return []

        query_text = f"{title} {content[:400]}"
        try:
            q_vec = self.tfidf_vectorizer.transform([query_text])
            sims = cosine_similarity(q_vec, self.corpus_tfidf).flatten()

            top_indices = np.argsort(sims)[::-1][:5]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if score > 0.05:
                    row = self.corpus_df.iloc[idx]
                    results.append({
                        "title": str(row.get("title", "")),
                        "source": str(row.get("source", "Báo chính thống")),
                        "url": str(row.get("url", "")),
                        "similarity": round(score, 3)
                    })
            return results
        except Exception:
            return []

    def _query_google_news_rss(self, title: str) -> List[Dict[str, Any]]:
        """Truy vấn tin tức xác thực thời gian thực qua Google News RSS tiếng Việt."""
        if not self.enable_web_search:
            return []

        query = self._clean_query_keywords(title)
        if not query:
            return []

        try:
            encoded_q = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=vi&gl=VN&ceid=VN:vi"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

            resp = requests.get(rss_url, headers=headers, timeout=self.web_search_timeout)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            items = root.findall(".//item")[:5]

            results = []
            for item in items:
                item_title = item.find("title").text if item.find("title") is not None else ""
                item_link = item.find("link").text if item.find("link") is not None else ""
                item_source = "Báo chí trực tuyến"
                if " - " in item_title:
                    parts = item_title.rsplit(" - ", 1)
                    item_title = parts[0].strip()
                    item_source = parts[1].strip()

                # Tính độ tương đồng từ vựng Jaccard cơ bản giữa 2 tiêu đề
                q_tokens = set(re.findall(r'\w+', query.lower()))
                cand_tokens = set(re.findall(r'\w+', item_title.lower()))
                jaccard = len(q_tokens & cand_tokens) / max(len(q_tokens | cand_tokens), 1)

                results.append({
                    "title": item_title,
                    "source": item_source,
                    "url": item_link,
                    "similarity": round(float(jaccard), 3)
                })
            return results
        except Exception:
            return []

    def verify_article(
        self,
        title: str,
        content: str,
        url: str = "",
        stylometric_vector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Trích xuất 4 đặc trưng kiểm chứng toán học và danh sách bằng chứng đối chứng.
        
        Trả về:
        - feature_vector: np.ndarray shape (4,) dtype float32
        - evidence_articles: List[Dict] danh sách bài báo đối chứng
        """
        title = title.strip() if isinstance(title, str) else ""
        content = content.strip() if isinstance(content, str) else ""

        evidence_list: List[Dict[str, Any]] = []

        # 1. Truy vấn kho cục bộ trước
        local_results = self._query_local_corpus(title, content)
        evidence_list.extend(local_results)

        # 2. Nếu điểm cao nhất của kho cục bộ thấp (< 0.40) và bật web search, gọi tìm kiếm online
        max_local_sim = max([r["similarity"] for r in local_results], default=0.0)
        if max_local_sim < 0.40 and self.enable_web_search:
            web_results = self._query_google_news_rss(title)
            evidence_list.extend(web_results)

        # Sắp xếp theo similarity giảm dần và lấy top 5
        evidence_list = sorted(evidence_list, key=lambda x: x["similarity"], reverse=True)[:5]

        # 3. Tính toán 4 đặc trưng kiểm chứng:
        if evidence_list:
            max_sim = float(evidence_list[0]["similarity"])
            # Đếm số lượng nguồn uy tín khác nhau có độ tương đồng >= 0.25
            verified_sources = {
                r["source"].lower() for r in evidence_list
                if r["similarity"] >= 0.25 and r["source"]
            }
            source_count = float(min(len(verified_sources), 5)) / 5.0
            
            # Tính độ tương đồng lập trường tiêu đề (stance)
            top_evidence_title = evidence_list[0]["title"]
            q_set = set(re.findall(r'\w+', title.lower()))
            e_set = set(re.findall(r'\w+', top_evidence_title.lower()))
            headline_stance = float(len(q_set & e_set) / max(len(q_set | e_set), 1))
        else:
            max_sim = 0.0
            source_count = 0.0
            headline_stance = 0.0

        # 4. Tính điểm phạt bất thường (Unverified Anomaly Penalty)
        # Nếu có dấu hiệu giật gân (viết hoa nhiều, dấu !, từ khóa clickbait) nhưng không có nguồn nào xác nhận
        sensational_cues = 0.0
        if stylometric_vector is not None and len(stylometric_vector) >= 6:
            # stylometric_vector: [upper_ratio, exclamation_density, ..., clickbait_hits, ...]
            upper_ratio = stylometric_vector[0]
            excl_density = stylometric_vector[1]
            clickbait_hits = stylometric_vector[5]
            sensational_cues = (upper_ratio * 1.5) + (excl_density * 2.0) + (min(clickbait_hits, 4) * 0.2)
        else:
            # Ước tính nhanh từ chuỗi văn bản nếu chưa có vector hình thức
            has_clickbait_words = bool(re.search(r'(chấn động|kinh hoàng|100%|lộ clip|thần kỳ|bài thuốc)', title, re.IGNORECASE))
            has_excl = "!" in title
            has_upper = any(w.isupper() and len(w) > 2 for w in title.split())
            sensational_cues = (0.4 if has_clickbait_words else 0.0) + (0.3 if has_excl else 0.0) + (0.3 if has_upper else 0.0)

        # Phạt nặng khi sensational_cues cao mà max_sim hoặc source_count = 0
        if max_sim < 0.25:
            anomaly_penalty = float(np.clip(sensational_cues * 0.8 + (1.0 - max_sim) * 0.3, 0.0, 1.0))
        else:
            anomaly_penalty = float(np.clip(sensational_cues * 0.2 * (1.0 - max_sim), 0.0, 0.5))

        feature_vector = np.array([
            np.clip(max_sim, 0.0, 1.0),
            np.clip(source_count, 0.0, 1.0),
            np.clip(headline_stance, 0.0, 1.0),
            np.clip(anomaly_penalty, 0.0, 1.0)
        ], dtype=np.float32)

        return feature_vector, evidence_list


if __name__ == "__main__":
    verifier = CrossSourceVerifier(enable_web_search=True)
    
    test_title_real = "Ngân hàng Nhà nước duy trì mặt bằng lãi suất điều hành ổn định"
    test_content_real = "Ngân hàng Nhà nước tiếp tục chỉ đạo các tổ chức tín dụng tiết giảm chi phí nhằm giảm lãi suất cho vay."
    vec_real, ev_real = verifier.verify_article(test_title_real, test_content_real)
    print("=== TEST REAL ARTICLE ===")
    print(f"Features: {vec_real}")
    print(f"Evidence count: {len(ev_real)}")
    for ev in ev_real[:2]:
        print(f" - [{ev['source']}] {ev['title']} (sim={ev['similarity']})")

    print("\n=== TEST FAKE ARTICLE ===")
    test_title_fake = "CHẤN ĐỘNG: Phát hiện phương thuốc nam chữa khỏi 100% ung thư trong 7 ngày!!!"
    test_content_fake = "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ khỏi mạng xã hội!"
    vec_fake, ev_fake = verifier.verify_article(test_title_fake, test_content_fake)
    print(f"Features: {vec_fake}")
    print(f"Evidence count: {len(ev_fake)}")
    for ev in ev_fake[:2]:
        print(f" - [{ev['source']}] {ev['title']} (sim={ev['similarity']})")
