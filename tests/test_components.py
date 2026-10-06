"""
Bộ kiểm thử tự động cho toàn bộ các module thành phần:
- TV2: StylometricExtractor (15 đặc trưng)
- TV3: PhoBertEmbedder (768 chiều L2)
- TV4: DualStreamClassifier (PyTorch forward) & FakeNewsPredictor (predict_with_shap)
- TV5: FastAPI server
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.features.stylometrics import StylometricExtractor, FEATURE_NAMES
from src.features.phobert_embedder import PhoBertEmbedder
from src.models.dual_stream_nn import DualStreamClassifier
from src.models.fusion_classifier import predict_with_shap
from src.api.server import app


def test_tv2_stylometric_extractor():
    """Kiểm tra trích xuất 15 đặc trưng của TV2."""
    extractor = StylometricExtractor()
    title = "CHẤN ĐỘNG: Lộ bài thuốc thần kỳ 100% sau 3 ngày!!!"
    content = "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ. Em thề cứu cả nhà đó."
    
    vec = extractor.extract_single(title, content)
    assert isinstance(vec, np.ndarray), "Kết quả phải là numpy ndarray"
    assert vec.shape == (15,), f"Kỳ vọng shape (15,), nhận được {vec.shape}"
    assert vec.dtype == np.float32, f"Kỳ vọng dtype np.float32, nhận được {vec.dtype}"
    assert not np.isnan(vec).any(), "Không được chứa giá trị NaN"
    assert not np.isinf(vec).any(), "Không được chứa giá trị Inf"


def test_tv3_phobert_embedder_mock():
    """Kiểm tra tạo vector nhúng 768 chiều L2 của TV3."""
    embedder = PhoBertEmbedder(lazy_load=True, mock_mode=True)
    title = "Tin tức thời sự hôm nay"
    content = "Bản tin thời sự cập nhật các vấn đề kinh tế xã hội trong nước."
    
    vec = embedder.encode_single_article(title, content)
    assert isinstance(vec, np.ndarray), "Kết quả phải là numpy ndarray"
    assert vec.shape == (768,), f"Kỳ vọng shape (768,), nhận được {vec.shape}"
    assert vec.dtype == np.float32, f"Kỳ vọng dtype np.float32, nhận được {vec.dtype}"
    
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0, f"Chuẩn L2 phải bằng 1.0, nhận được {norm}"


def test_tv4_dual_stream_nn():
    """Kiểm tra forward pass mạng nơ-ron đa luồng PyTorch của TV4."""
    model = DualStreamClassifier()
    model.eval()
    
    batch_size = 3
    dummy_sem = torch.randn(batch_size, 768)
    dummy_sty = torch.randn(batch_size, 15)
    
    with torch.no_grad():
        out = model(dummy_sem, dummy_sty)
        
    assert out.shape == (batch_size, 1), f"Kỳ vọng output shape ({batch_size}, 1), nhận được {out.shape}"
    # Xác suất trong khoảng [0, 1]
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Giá trị xác suất phải thuộc [0, 1]"


def test_tv4_predict_with_shap():
    """Kiểm tra hàm hợp nhất và giải thích SHAP của TV4."""
    title = "CHẤN ĐỘNG: Bài thuốc thần kỳ chữa dứt điểm ung thư 100%!!!"
    content = "Bà con ơi chia sẻ ngay kẻo xóa!"
    url = "https://sample-fake-news.vn"
    
    res = predict_with_shap(title, content, url)
    assert res["status"] == "success"
    assert res["prediction"] in ["TIN GIẢ / GIẬT GÂN", "TIN THẬT / XÁC THỰC"]
    assert 0.0 <= res["probability_fake"] <= 1.0
    assert 0.0 <= res["confidence_score"] <= 100.0
    assert res["risk_tier"] in ["CRITICAL", "MODERATE", "LOW"]
    assert len(res["top_shap_factors"]) > 0
    assert "stylometric_summary" in res


def test_tv5_fastapi_server():
    """Kiểm tra dịch vụ REST API FastAPI của TV5."""
    client = TestClient(app)
    
    # 1. Health check
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"
    
    # 2. Predict endpoint
    res_p = client.post("/api/v1/predict", json={
        "title": "Bản tin kinh tế tài chính",
        "content": "Chỉ số giá tiêu dùng tháng này duy trì ổn định theo báo cáo của Tổng cục Thống kê.",
        "url": "https://chinhthong.vn/tin-kinh-te"
    })
    assert res_p.status_code == 200
    body = res_p.json()
    assert body["status"] == "success"
    assert "prediction" in body
    assert "top_shap_factors" in body
