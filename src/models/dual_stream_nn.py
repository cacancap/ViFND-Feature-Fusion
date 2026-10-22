"""
ViFND - Vietnamese Fake News Detection
Module: src/models/dual_stream_nn.py
Vai trò: Thành viên 4 (TV4) - Mô hình hóa Mạng Nơ-ron Đa Nhánh (Multi-Stream PyTorch)

Kiến trúc mạng Late Fusion xử lý bất đối xứng chiều giữa:
- Nhánh Ngữ nghĩa (Semantic Branch): 768 -> 128 (BatchNorm, ReLU, Dropout)
- Nhánh Hình thức (Stylometric Branch): 15 -> 32 (BatchNorm, ReLU, Dropout)
- Nhánh Kiểm chứng chéo (Verification Branch): 4 -> 16 (BatchNorm, ReLU, Dropout)
- Tầng Hợp nhất (Fusion Classifier): (128 + 32 + 16 = 176) -> 64 -> 1 (Sigmoid)
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
from typing import Dict, Any, Optional


class DualStreamClassifier(nn.Module):
    """
    Mạng nơ-ron sâu hợp nhất muộn đa luồng (Multi-Stream Late Fusion Neural Network).
    - sem_dim: 768 (từ PhoBERT mean-pooled)
    - sty_dim: 15 (từ StylometricExtractor)
    - veri_dim: 4 (từ CrossSourceVerifier)
    """

    def __init__(
        self,
        sem_dim: int = 768,
        sty_dim: int = 15,
        veri_dim: int = 4,
        dropout_p: float = 0.3
    ):
        super().__init__()
        self.veri_dim = veri_dim
        
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

        # Nhánh 3: Nhánh kiểm chứng chéo đa nguồn 4 -> 16
        if veri_dim > 0:
            self.veri_branch = nn.Sequential(
                nn.Linear(veri_dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            fusion_in = 128 + 32 + 16
        else:
            self.veri_branch = None
            fusion_in = 128 + 32
        
        # Tầng Hợp nhất & Phân loại nhị phân
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        sem_x: torch.Tensor,
        sty_x: torch.Tensor,
        veri_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        sem_x: Tensor shape (Batch, 768)
        sty_x: Tensor shape (Batch, 15)
        veri_x: Tensor shape (Batch, 4) - nếu không truyền sẽ tự động bù 0 (fallback an toàn)
        Trả về xác suất tin giả: Tensor shape (Batch, 1) trong khoảng [0, 1].
        """
        h_sem = self.sem_branch(sem_x)
        h_sty = self.sty_branch(sty_x)
        
        if self.veri_branch is not None:
            if veri_x is None:
                veri_x = torch.zeros(sem_x.size(0), self.veri_dim, device=sem_x.device)
            h_veri = self.veri_branch(veri_x)
            fused = torch.cat([h_sem, h_sty, h_veri], dim=1)
        else:
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
    model = DualStreamClassifier(veri_dim=4)
    model.eval()
    
    batch_size = 4
    dummy_sem = torch.randn(batch_size, 768)
    dummy_sty = torch.randn(batch_size, 15)
    dummy_veri = torch.randn(batch_size, 4)
    
    with torch.no_grad():
        out_triple = model(dummy_sem, dummy_sty, dummy_veri)
        out_dual = model(dummy_sem, dummy_sty)
        
    summary = get_model_summary(model)
    print("Multi-Stream Classifier PyTorch khởi tạo thành công:")
    print(f"  - Tổng tham số: {summary['total_parameters']:,}")
    print(f"  - Output shape (Triple Stream): {out_triple.shape}")
    print(f"  - Output shape (Dual Stream fallback): {out_dual.shape}")
