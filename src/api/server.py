"""
ViFND - Vietnamese Fake News Detection
Module: src/api/server.py
Vai trò: Thành viên 5 (TV5) - REST API Microservice (FastAPI)

Cung cấp các endpoints chuẩn RESTful phục vụ tích hợp hệ thống bên thứ ba và Web UI:
- GET  /health          : Kiểm tra trạng thái hệ thống
- POST /api/v1/predict  : Phân loại tin tức, trả về độ tin cậy và bằng chứng đối chứng
- POST /api/v1/explain  : Phân tích chi tiết các trọng số SHAP giải thích mô hình
"""

import sys
from pathlib import Path

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.models.fusion_classifier import predict_with_shap

app = FastAPI(
    title="ViFND API - Vietnamese Fake News Detection Service",
    description="REST API cho hệ thống phát hiện tin giả tiếng Việt qua Hợp nhất Đặc trưng Ba Luồng và XAI",
    version="1.1.0"
)


class ArticleRequest(BaseModel):
    title: str = Field(
        ...,
        description="Tiêu đề bài báo tiếng Việt",
        json_schema_extra={"example": "CHẤN ĐỘNG: Lộ bài thuốc thần kỳ chữa khỏi ung thư trong 3 ngày!!!"}
    )
    content: str = Field(
        ...,
        description="Thân bài hoặc đoạn trích nội dung",
        json_schema_extra={"example": "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ! Em thề thuốc tiên này cứu cả nhà đó."}
    )
    url: Optional[str] = Field(
        default="",
        description="Đường dẫn nguồn bài báo nếu có",
        json_schema_extra={"example": "https://sample-news.vn/post/102"}
    )


class ShapFactor(BaseModel):
    feature: str
    contribution: float
    direction: str
    description_vi: str


class PredictionResponse(BaseModel):
    status: str
    url: Optional[str] = ""
    title: str
    prediction: str
    probability_fake: float
    confidence_score: float
    risk_tier: str
    stylometric_summary: Dict[str, Any]
    cross_verification_summary: Optional[Dict[str, Any]] = None
    evidence_articles: Optional[List[Dict[str, Any]]] = None
    top_shap_factors: List[ShapFactor]


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Kiểm tra độ sẵn sàng của dịch vụ."""
    return {
        "status": "healthy",
        "service": "ViFND Detection Microservice",
        "version": "1.1.0",
        "engine": "Tri-Stream Feature Fusion (PhoBERT + Stylometrics + Cross-Source Verifier + XGBoost)"
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_article(request: ArticleRequest):
    """
    Phân loại bài báo, kiểm chứng đối chiếu và trích xuất lý do giải thích SHAP.
    """
    if not request.title.strip() and not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tiêu đề hoặc nội dung bài báo không được để trống."
        )

    try:
        result = predict_with_shap(request.title, request.content, request.url or "")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi phân tích dự đoán: {str(e)}"
        )


@app.post("/api/v1/explain", tags=["Explainability"])
def explain_article(request: ArticleRequest):
    """
    Trả về chỉ riêng phần giải thích mô hình XAI (SHAP) và bằng chứng kiểm chứng.
    """
    res = predict_article(request)
    return {
        "status": res.status,
        "prediction": res.prediction,
        "risk_tier": res.risk_tier,
        "top_shap_factors": res.top_shap_factors,
        "cross_verification_summary": res.cross_verification_summary,
        "evidence_articles": res.evidence_articles
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
