"""
RSS Feeds Crawler & Scraper Module for TV1
Collects real news samples from authoritative Vietnamese news portals (VnExpress, Tuoi Tre, VietnamNet).
"""
import os
import sys
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

RSS_FEEDS = [
    ("https://vnexpress.net/rss/thoi-su.rss", "VnExpress_ThoiSu"),
    ("https://vnexpress.net/rss/the-gioi.rss", "VnExpress_TheGioi"),
    ("https://vnexpress.net/rss/kinh-doanh.rss", "VnExpress_KinhDoanh"),
    ("https://vnexpress.net/rss/khoa-hoc.rss", "VnExpress_KhoaHoc"),
    ("https://tuoitre.vn/rss/thoi-su.rss", "TuoiTre_ThoiSu"),
    ("https://tuoitre.vn/rss/the-gioi.rss", "TuoiTre_TheGioi"),
    ("https://vietnamnet.vn/rss/thoi-su.rss", "VietnamNet_ThoiSu")
]

def fetch_rss_feed(feed_url: str, source_name: str, max_items: int = 50) -> list:
    articles = []
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:max_items]:
                title = item.findtext("title", "").strip()
                desc_raw = item.findtext("description", "").strip()
                desc_clean = BeautifulSoup(desc_raw, "html.parser").get_text().strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                
                if len(title) > 10 and len(desc_clean) > 20:
                    articles.append({
                        "title": title,
                        "content": desc_clean,
                        "source": source_name,
                        "url": link,
                        "pub_date": pub_date,
                        "label": 0 # Real
                    })
    except Exception as e:
        print(f"[Warning] Failed to fetch {feed_url}: {e}")
    return articles

def crawl_rss_dataset(output_path: str = "data/raw/rss_crawled_news.parquet") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_articles = []
    
    for url, name in RSS_FEEDS:
        print(f"[TV1 Crawler] Fetching {name} ({url})...")
        items = fetch_rss_feed(url, name, max_items=40)
        all_articles.extend(items)
        time.sleep(1.0) # Polite delay
        
    df = pd.DataFrame(all_articles)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    print(f"[TV1 Crawler] Crawled {len(df)} articles saved to: {output_path}")
    return df

if __name__ == "__main__":
    crawl_rss_dataset()
