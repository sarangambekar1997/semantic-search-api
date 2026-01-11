import pytest
import numpy as np
from app.embeddings import generate_embedding  # Adjust import path

def test_generate_embedding_returns_array():
    """Test embedding generation returns numpy array"""
    embedding = generate_embedding("urgent payment")
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0

def test_embedding_dimensions_consistent():
    """Test embeddings have consistent dimensions"""
    emb1 = generate_embedding("payment issue")
    emb2 = generate_embedding("urgent payment")
    assert emb1.shape == emb2.shape
    assert len(emb1) == len(emb2)

def test_embedding_normalized():
    """Test embeddings are properly normalized"""
    embedding = generate_embedding("test query")
    norm = np.linalg.norm(embedding)
    assert pytest.approx(1.0, abs=0.1) == norm