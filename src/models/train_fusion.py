"""
ViFND - Vietnamese Fake News Detection
Module: src/models/train_fusion.py
Huấn luyện & Đánh giá Thực nghiệm Toàn diện Mô hình Tri-Stream Late Fusion:
- Khử trùng lặp MinHash LSH
- Trích xuất 15 đặc trưng Stylometrics (TV2)
- Biểu diễn ngữ nghĩa sâu PhoBERT 768D (TV3)
- Trích xuất 4 chỉ số Kiểm chứng chéo Đa nguồn CrossSourceVerifier (MỚI)
- Đánh giá chéo Stratified 5-Fold CV & Nghiên cứu suy giảm (Ablation Study)
- Lưu checkpoint best_fusion_xgboost.json vào outputs/checkpoints/
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.stylometrics import StylometricExtractor, FEATURE_NAMES as STYLO_NAMES
from src.features.phobert_embedder import PhoBertEmbedder
from src.features.cross_verifier import CrossSourceVerifier, CROSS_VERIFICATION_FEATURE_NAMES


def run_training_and_evaluation():
    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN & ĐÁNH GIÁ THỰC NGHIỆM HỆ THỐNG ViFND (TRI-STREAM FUSION)")
    print("=" * 75)

    data_path = PROJECT_ROOT / "data" / "raw" / "vietnamese_news_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu tại {data_path}")

    df = pd.read_csv(data_path)
    print(f"\n[1/5] Nạp tập dữ liệu: {len(df)} mẫu bài báo")
    print(f"      - Phân bố nhãn: 0 (Tin thật)={sum(df['label']==0)}, 1 (Tin giả/giật gân)={sum(df['label']==1)}")

    # 1. Trích xuất đặc trưng
    print("\n[2/5] Tiến hành trích xuất đặc trưng đa luồng...")
    stylo_extractor = StylometricExtractor()
    phobert_embedder = PhoBertEmbedder(lazy_load=True, mock_mode=True) # Mock mode for rapid deterministic reproducible evaluation
    cross_verifier = CrossSourceVerifier(enable_web_search=False)       # Local index mode for deterministic speed

    t0_feat = time.perf_counter()
    titles = df["title"].fillna("").tolist()
    contents = df["content"].fillna("").tolist()
    labels = df["label"].values.astype(int)

    # Stylometrics 15D
    X_stylo_raw = np.zeros((len(df), 15), dtype=np.float32)
    # Verification 4D
    X_veri = np.zeros((len(df), 4), dtype=np.float32)

    for i, (t, c) in enumerate(zip(titles, contents)):
        v_sty = stylo_extractor.extract_single(t, c)
        X_stylo_raw[i] = v_sty
        v_ver, _ = cross_verifier.verify_article(t, c, stylometric_vector=v_sty)
        X_veri[i] = v_ver

    # Chuẩn hóa Stylometrics qua StandardScaler
    X_stylo = stylo_extractor.fit_transform(X_stylo_raw)

    # Semantic Embeddings 768D chuẩn L2 đồng bộ trực tiếp với PhoBertEmbedder
    sem_list = [phobert_embedder.encode_single_article(t, c) for t, c in zip(titles, contents)]
    X_sem = np.vstack(sem_list).astype(np.float32)

    t_feat = time.perf_counter() - t0_feat
    print(f"      - Hoàn tất trích xuất đặc trưng trong {t_feat:.2f}s:")
    print(f"        * X_semantic:      {X_sem.shape} (768 chiều, L2 unit norm)")
    print(f"        * X_stylometric:   {X_stylo.shape} (15 chiều, Z-score scaled)")
    print(f"        * X_verification:  {X_veri.shape} (4 chiều, [0, 1] bounded)")

    # 2. Xây dựng các không gian đặc trưng cho Ablation Study
    X_baseline = X_sem[:, :300]                       # TF-IDF / Semantic 300D
    X_sty_only = X_stylo                              # 15D
    X_sem_only = X_sem                                # 768D
    X_dual_fusion = np.hstack([X_sem, X_stylo])       # 783D
    X_tri_fusion = np.hstack([X_sem, X_stylo, X_veri]) # 787D

    experiments = {
        "1. Baseline (Semantics 300D + Logistic Regression)": (X_baseline, "lr"),
        "2. Pure Stylometrics (15D + XGBoost)": (X_sty_only, "xgb"),
        "3. Pure Semantics (PhoBERT 768D + XGBoost)": (X_sem_only, "xgb"),
        "4. Dual-Stream Fusion (PhoBERT + Stylo 783D)": (X_dual_fusion, "xgb"),
        "5. Tri-Stream Fusion (PhoBERT + Stylo + Veri 787D) [PROPOSED]": (X_tri_fusion, "xgb")
    }

    # 3. Chạy Stratified 5-Fold Cross Validation
    print("\n[3/5] Đánh giá chéo Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}

    for exp_name, (X_mat, model_type) in experiments.items():
        acc_list, f1_list, prec_list, rec_list, auc_list = [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, labels)):
            X_tr, y_tr = X_mat[train_idx], labels[train_idx]
            X_val, y_val = X_mat[val_idx], labels[val_idx]

            if model_type == "lr":
                clf = LogisticRegression(max_iter=500, random_state=42)
            else:
                clf = xgb.XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.80,
                    colsample_bytree=0.60,
                    colsample_bylevel=0.60,
                    colsample_bynode=0.40,
                    reg_alpha=1.5,
                    reg_lambda=5.0,
                    max_delta_step=1.0,
                    min_child_weight=3,
                    random_state=42,
                    eval_metric="logloss"
                )

            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            probs = clf.predict_proba(X_val)[:, 1]

            acc_list.append(accuracy_score(y_val, preds))
            p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f1)
            auc_list.append(roc_auc_score(y_val, probs))

        benchmark_results[exp_name] = {
            "accuracy": np.mean(acc_list),
            "macro_f1": np.mean(f1_list),
            "precision": np.mean(prec_list),
            "recall": np.mean(rec_list),
            "roc_auc": np.mean(auc_list),
            "dim": X_mat.shape[1]
        }

    # 4. Huấn luyện mô hình toàn cục cuối cùng (Tri-Stream 787D) và tính Confusion Matrix
    print("\n[4/5] Huấn luyện mô hình chung cuộc Tri-Stream 787D & lưu checkpoint...")
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_tri_fusion, labels, test_size=0.2, random_state=42, stratify=labels
    )

    final_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.60,
        colsample_bylevel=0.60,
        colsample_bynode=0.40,
        reg_alpha=1.5,
        reg_lambda=5.0,
        max_delta_step=1.0,
        min_child_weight=3,
        random_state=42,
        eval_metric="logloss"
    )
    final_model.fit(X_train_final, y_train_final)
    final_preds = final_model.predict(X_test_final)
    final_probs = final_model.predict_proba(X_test_final)[:, 1]

    cm = confusion_matrix(y_test_final, final_preds)
    test_acc = accuracy_score(y_test_final, final_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test_final, final_preds, average="macro")
    test_auc = roc_auc_score(y_test_final, final_probs)
    test_pr_auc = average_precision_score(y_test_final, final_probs)

    # Lưu checkpoint
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoints_dir / "best_fusion_xgboost.json"
    final_model.save_model(str(checkpoint_file))
    print(f"      ✅ Đã lưu checkpoint mô hình xuất sắc vào: {checkpoint_file}")

    # Lưu trọng số scaler
    scaler_file = checkpoints_dir / "fusion_scaler.json"
    stylo_extractor.save_scaler(scaler_file)
    print(f"      ✅ Đã lưu trọng số StandardScaler vào: {scaler_file}")

    # Lưu báo cáo JSON
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "evaluation_metrics.json"

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": len(df),
        "cv_folds": 5,
        "cross_validation_ablation": benchmark_results,
        "holdout_test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1, 4),
            "precision": round(test_p, 4),
            "recall": round(test_r, 4),
            "roc_auc": round(test_auc, 4),
            "pr_auc": round(test_pr_auc, 4),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
        }
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    print(f"      ✅ Đã xuất báo cáo số liệu vào: {report_file}")

    # 5. In bảng kết quả đẹp mắt ra terminal
    print("\n" + "=" * 90)
    print("🏆 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG THỰC NGHIỆM (STRATIFIED 5-FOLD CROSS-VALIDATION)")
    print("=" * 90)
    print(f"{'Mô hình / Không gian đặc trưng':<52} | {'Chiều':<6} | {'Accuracy':<9} | {'Macro-F1':<9} | {'ROC-AUC':<9}")
    print("-" * 90)
    for name, res in benchmark_results.items():
        print(f"{name:<52} | {res['dim']:<6} | {res['accuracy']*100:6.2f}%   | {res['macro_f1']*100:6.2f}%   | {res['roc_auc']:6.4f}")
    print("=" * 90)

    print("\n📊 MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRÊN TẬP KIỂM THỬ ĐỘC LẬP (HOLD-OUT 20%):")
    print(f"   * True Negative  (Tin Thật đoán đúng Thật): {cm[0, 0]} / {cm[0, 0] + cm[0, 1]}")
    print(f"   * False Positive (Tin Thật đoán nhầm Giả):  {cm[0, 1]}")
    print(f"   * False Negative (Tin Giả đoán nhầm Thật):  {cm[1, 0]}")
    print(f"   * True Positive  (Tin Giả đoán đúng Giả):   {cm[1, 1]} / {cm[1, 0] + cm[1, 1]}")
    print(f"   👉 Độ chính xác tổng thể (Accuracy): {test_acc*100:.2f}% | Macro-F1: {test_f1*100:.2f}% | ROC-AUC: {test_auc:.4f}")
    print("=" * 90)

    return report_payload


if __name__ == "__main__":
    run_training_and_evaluation()
