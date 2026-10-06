# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860

WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio fastapi uvicorn

# Sao chép toàn bộ mã nguồn
COPY . /app

# Mở cổng Gradio UI (7860) và FastAPI (8000)
EXPOSE 7860 8000

# Lệnh mặc định chạy giao diện Web Gradio
CMD ["python", "app.py"]
