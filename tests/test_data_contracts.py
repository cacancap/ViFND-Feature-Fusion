"""
Automated Contract Verification Test Suite for ViFND-Feature-Fusion
Verifies that all data artifacts, fixtures, and interfaces adhere strictly to team specifications.
Run with: pytest tests/test_data_contracts.py
"""
import os
import json
import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../data/fixtures")

def test_contract_p1_corpus_schema():
    """Verifies Contract 1: Corpus Data Schema (TV1 -> TV2, TV3)"""
    parquet_path = os.path.join(FIXTURES_DIR, "sample_corpus.parquet")
    assert os.path.exists(parquet_path), f"Contract 1 Violation: {parquet_path} does not exist."
    
    df = pd.read_parquet(parquet_path)
    required_cols = ["doc_id", "title", "content", "source", "url", "label"]
    for col in required_cols:
        assert col in df.columns, f"Contract 1 Violation: Missing required column '{col}'."
        
    assert len(df) > 0, "Contract 1 Violation: Corpus must contain at least 1 row."
    assert df["title"].isnull().sum() == 0, "Contract 1 Violation: Found nulls in 'title'."
    assert df["content"].isnull().sum() == 0, "Contract 1 Violation: Found nulls in 'content'."
    assert set(df["label"].unique()).issubset({0, 1}), "Contract 1 Violation: 'label' must be 0 or 1."

def test_contract_p2_stylometrics_tensor():
    """Verifies Contract 2: Stylometric Feature Tensor (TV2 -> TV4)"""
    npy_path = os.path.join(FIXTURES_DIR, "sample_X_stylometric.npy")
    names_path = os.path.join(FIXTURES_DIR, "stylometric_feature_names.json")
    
    assert os.path.exists(npy_path), f"Contract 2 Violation: {npy_path} does not exist."
    assert os.path.exists(names_path), f"Contract 2 Violation: {names_path} does not exist."
    
    X_sty = np.load(npy_path)
    assert X_sty.ndim == 2, "Contract 2 Violation: Stylometric tensor must be 2D."
    assert X_sty.shape[1] == 15, f"Contract 2 Violation: Expected exactly 15 features, got {X_sty.shape[1]}."
    assert X_sty.dtype == np.float32, f"Contract 2 Violation: Expected float32, got {X_sty.dtype}."
    assert not np.isnan(X_sty).any(), "Contract 2 Violation: Tensor contains NaN."
    assert not np.isinf(X_sty).any(), "Contract 2 Violation: Tensor contains Inf."
    
    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    assert len(names) == 15, "Contract 2 Violation: Feature names list must have 15 entries."

def test_contract_p3_semantic_embeddings():
    """Verifies Contract 3: Semantic Embeddings Tensor (TV3 -> TV4)"""
    npy_path = os.path.join(FIXTURES_DIR, "sample_X_semantic.npy")
    assert os.path.exists(npy_path), f"Contract 3 Violation: {npy_path} does not exist."
    
    X_sem = np.load(npy_path)
    assert X_sem.ndim == 2, "Contract 3 Violation: Semantic tensor must be 2D."
    assert X_sem.shape[1] == 768, f"Contract 3 Violation: Expected 768 dimensions for PhoBERT, got {X_sem.shape[1]}."
    assert X_sem.dtype == np.float32, f"Contract 3 Violation: Expected float32, got {X_sem.dtype}."
    assert not np.isnan(X_sem).any(), "Contract 3 Violation: Tensor contains NaN."
    
    # Check L2 unit norm
    norms = np.linalg.norm(X_sem, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Contract 3 Violation: Vectors must be L2 unit normalized."

def test_contract_p4_prediction_schema():
    """Verifies Contract 4: Prediction & XAI Output Schema (TV4 -> TV5)"""
    json_path = os.path.join(FIXTURES_DIR, "sample_prediction_output.json")
    assert os.path.exists(json_path), f"Contract 4 Violation: {json_path} does not exist."
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    required_keys = ["status", "prediction", "probability_fake", "risk_tier", "top_shap_factors"]
    for key in required_keys:
        assert key in data, f"Contract 4 Violation: Missing key '{key}' in prediction output."
        
    assert 0.0 <= data["probability_fake"] <= 1.0, "Contract 4 Violation: probability_fake must be in [0, 1]."
    assert isinstance(data["top_shap_factors"], list), "Contract 4 Violation: top_shap_factors must be a list."
    assert len(data["top_shap_factors"]) > 0, "Contract 4 Violation: top_shap_factors cannot be empty."
    
    first_factor = data["top_shap_factors"][0]
    for k in ["feature", "contribution", "description_vi"]:
        assert k in first_factor, f"Contract 4 Violation: Missing factor key '{k}'."
