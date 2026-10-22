"""
Bộ kiểm thử tự động cho toàn bộ các module thành phần:
- TV2: StylometricExtractor (15 đặc trưng)
- TV3: PhoBertEmbedder (768 chiều L2)
- TV4: CrossSourceVerifier (4 đặc trưng kiểm chứng đa nguồn & danh sách bằng chứng) [MỚI]
- TV4: DualStreamClassifier / Multi-Stream PyTorch (forward pass) & FakeNewsPredictor (predict_with_shap)
- TV5: FastAPI server (/health, /api/v1/predict)
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
from src.features.cross_verifier import CrossSourceVerifier, CROSS_VERIFICATION_FEATURE_NAMES
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


def test_tv4_cross_source_verifier():
    """Kiểm tra Tầng kiểm chứng chéo đa nguồn (CrossSourceVerifier) [MỚI]."""
    verifier = CrossSourceVerifier(enable_web_search=False) # Chạy offline mode
    title = "Ngân hàng Nhà nước giữ vững ổn định tỷ giá và thanh khoản hệ thống tín dụng"
    content = "Theo báo cáo, thanh khoản hệ thống tín dụng tiếp tục duy trì trạng thái tích cực."
    
    vec, evidence = verifier.verify_article(title, content)
    assert isinstance(vec, np.ndarray), "Vector kiểm chứng phải là numpy ndarray"
    assert vec.shape == (4,), f"Kỳ vọng shape (4,), nhận được {vec.shape}"
    assert vec.dtype == np.float32, f"Kỳ vọng float32, nhận được {vec.dtype}"
    assert (vec >= 0.0).all() and (vec <= 1.0).all(), "Các giá trị đặc trưng kiểm chứng phải thuộc [0, 1]"
    assert isinstance(evidence, list), "Danh sách bằng chứng phải là list"


def test_tv4_dual_stream_nn():
    """Kiểm tra forward pass mạng nơ-ron đa luồng PyTorch của TV4."""
    model = DualStreamClassifier(veri_dim=4)
    model.eval()
    
    batch_size = 3
    dummy_sem = torch.randn(batch_size, 768)
    dummy_sty = torch.randn(batch_size, 15)
    dummy_veri = torch.randn(batch_size, 4)
    
    with torch.no_grad():
        out_triple = model(dummy_sem, dummy_sty, dummy_veri)
        out_fallback = model(dummy_sem, dummy_sty)
        
    assert out_triple.shape == (batch_size, 1), f"Kỳ vọng output shape ({batch_size}, 1)"
    assert (out_triple >= 0.0).all() and (out_triple <= 1.0).all(), "Xác suất phải thuộc [0, 1]"
    assert out_fallback.shape == (batch_size, 1), "Fallback dual-stream vẫn phải hoạt động bình thường"


def test_tv4_predict_with_shap():
    """Kiểm tra hàm hợp nhất và giải thích SHAP của TV4 có kèm tầng kiểm chứng."""
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
    assert "cross_verification_summary" in res
    assert "evidence_articles" in res
    assert isinstance(res["evidence_articles"], list)


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
