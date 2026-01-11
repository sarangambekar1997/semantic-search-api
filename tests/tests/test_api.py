import pytest
from fastapi import status

def test_semantic_search_valid_query(client, sample_query):
    """Test semantic search with valid query"""
    response = client.post(
        "/search/semantic",
        json={"query": sample_query, "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    assert "record" in data[0]
    assert "score" in data[0]
    assert data[0]["score"] > 0

def test_semantic_search_top_k(client):
    """Test different top_k values"""
    response = client.post(
        "/search/semantic",
        json={"query": "payment", "top_k": 3}
    )
    assert response.status_code == 200
    assert len(response.json()) <= 3

def test_semantic_search_missing_query(client):
    """Test missing query parameter"""
    response = client.post(
        "/search/semantic",
        json={"top_k": 5}
    )
    assert response.status_code == 422  # Validation error

def test_semantic_search_invalid_top_k(client):
    """Test invalid top_k value"""
    response = client.post(
        "/search/semantic",
        json={"query": "test", "top_k": -1}
    )
    assert response.status_code == 422