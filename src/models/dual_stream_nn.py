"""
ViFND - Vietnamese Fake News Detection
Module: src/models/dual_stream_nn.py
Vai trò: Thành viên 4 (TV4) - Mô hình hóa Mạng Nơ-ron Đa Nhánh (Dual-Stream PyTorch)

Kiến trúc mạng Late Fusion xử lý bất đối xứng chiều giữa ngữ nghĩa (768D) và hình thức (15D):
- Nhánh Ngữ nghĩa (Semantic Branch): 768 -> 128 (BatchNorm, ReLU, Dropout)
- Nhánh Hình thức (Stylometric Branch): 15 -> 32 (BatchNorm, ReLU, Dropout)
- Tầng Hợp nhất (Fusion Classifier): (128 + 32) -> 64 -> 1 (Sigmoid)
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from typing import Dict, Any


class DualStreamClassifier(nn.Module):
    """
    Mạng nơ-ron sâu hợp nhất muộn 2 luồng (Dual-Stream Late Fusion Neural Network).
    - sem_dim: 768 (từ PhoBERT mean-pooled)
    - sty_dim: 15 (từ StylometricExtractor)
    """

    def __init__(self, sem_dim: int = 768, sty_dim: int = 15, dropout_p: float = 0.3):
        super().__init__()
        
        # Nhánh 1: Nén biểu diễn ngữ nghĩa dày đặc 768 -> 128
        self.sem_branch = nn.Sequential(
            nn.Linear(sem_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
        # Nhánh 2: Mở rộng đặc trưng phong cách hình thức 15 -> 32
        self.sty_branch = nn.Sequential(
            nn.Linear(sty_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(max(0.1, dropout_p - 0.1))
        )
        
        # Tầng Hợp nhất & Phân loại nhị phân
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sem_x: torch.Tensor, sty_x: torch.Tensor) -> torch.Tensor:
        """
        sem_x: Tensor shape (Batch, 768)
        sty_x: Tensor shape (Batch, 15)
        Trả về xác suất tin giả: Tensor shape (Batch, 1) trong khoảng [0, 1].
        """
        # Nếu batch_size = 1 và BatchNorm1d gặp lỗi, chuyển sang eval mode
        h_sem = self.sem_branch(sem_x)
        h_sty = self.sty_branch(sty_x)
        fused = torch.cat([h_sem, h_sty], dim=1)
        prob = self.classifier(fused)
        return prob


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Trả về số lượng tham số có thể huấn luyện."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }


if __name__ == "__main__":
    model = DualStreamClassifier()
    model.eval()
    
    # Kiểm thử nhanh với dữ liệu ngẫu nhiên
    batch_size = 4
    dummy_sem = torch.randn(batch_size, 768)
    dummy_sty = torch.randn(batch_size, 15)
    
    with torch.no_grad():
        out = model(dummy_sem, dummy_sty)
        
    summary = get_model_summary(model)
    print("DualStreamClassifier PyTorch khởi tạo thành công:")
    print(f"  - Tổng tham số: {summary['total_parameters']:,}")
    print(f"  - Tham số huấn luyện: {summary['trainable_parameters']:,}")
    print(f"  - Output shape: {out.shape}")
    print(f"  - Giá trị mẫu: {out.squeeze().tolist()}")
