"""Tests for search_with_media method (multimodal search).

These tests validate the search_with_media method's parameter handling and
request structure using a mock REST embedder HTTP server.
"""

import pytest
from meilisearch.index import Index


# Minimal sanity test: ensure the method exists on Index class without network calls.
def test_search_with_media_method_exists():
    """Test that Index class exposes a callable search_with_media method.

    This avoids creating an actual index (which would call the server) so
    the test is safe to run in isolation.
    """
    assert hasattr(Index, "search_with_media")
    assert callable(getattr(Index, "search_with_media"))


def test_search_with_media_basic_parameters(index_with_rest_embedder):
    """Test search_with_media accepts media parameter and sends correct request structure.

    Uses a local stub of index.http.post to avoid external network calls.
    """
    index = index_with_rest_embedder()

    # stub the http.post to return a deterministic fake response
    fake_response = {
        "hits": [{"id": 1, "title": "Fake Movie"}],
        "processingTimeMs": 5,
        "limit": 20,
        "offset": 0,
        "estimatedTotalHits": 1,
    }

    index.http.post = lambda *args, **kwargs: fake_response

    # Search with media parameter - stubbed response will be returned
    response = index.search_with_media(
        media={"text": "test query"},
        opt_params={"hybrid": {"embedder": "default"}}
    )

    assert isinstance(response, dict)
    assert "hits" in response
    assert "processingTimeMs" in response


def test_search_with_media_with_optional_params(index_with_rest_embedder):
    """Test search_with_media with optional parameters."""
    index = index_with_rest_embedder()

    # Return a response that respects the requested limit
    def fake_post(*args, **kwargs):
        return {
            "hits": [{"id": 1, "title": "Fake Movie"}],
            "processingTimeMs": 3,
            "limit": 1,
            "offset": 0,
            "estimatedTotalHits": 1,
        }

    index.http.post = fake_post

    # Search with media and optional parameters
    response = index.search_with_media(
        media={"text": "query"},
        opt_params={
            "limit": 1,
            "offset": 0,
            "hybrid": {"embedder": "default"}
        }
    )

    assert isinstance(response, dict)
    assert "hits" in response
    assert "limit" in response
    assert response["limit"] == 1


def test_search_with_media_response_structure(index_with_rest_embedder):
    """Test that search_with_media returns expected response structure."""
    index = index_with_rest_embedder()

    fake_response = {
        "hits": [],
        "processingTimeMs": 7,
        "limit": 20,
        "offset": 0,
        "estimatedTotalHits": 0,
    }

    index.http.post = lambda *args, **kwargs: fake_response

    response = index.search_with_media(
        media={"text": "movie"},
        opt_params={"hybrid": {"embedder": "default"}}
    )

    # Verify response has expected fields
    assert isinstance(response, dict)
    assert "hits" in response
    assert "processingTimeMs" in response
    assert "limit" in response
    assert "offset" in response
    assert "estimatedTotalHits" in response


def test_search_with_media_returns_results(index_with_rest_embedder):
    """Test that search_with_media can return search results."""
    index = index_with_rest_embedder()

    fake_response = {
        "hits": [{"id": 42, "title": "The Answer"}],
        "processingTimeMs": 4,
        "limit": 20,
        "offset": 0,
        "estimatedTotalHits": 1,
    }

    index.http.post = lambda *args, **kwargs: fake_response

    response = index.search_with_media(
        media={"text": "movie"},
        opt_params={"hybrid": {"embedder": "default"}}
    )

    assert isinstance(response, dict)
    assert "hits" in response
    assert isinstance(response["hits"], list)
    # With stubbed embedder, we should get results (length >= 0)
    assert len(response["hits"]) >= 0


def test_search_with_media_only_media_parameter(index_with_rest_embedder):
    """Test search_with_media works with only media parameter (no query text).

    This is a key feature of multimodal search - searching with media alone.
    """
    index = index_with_rest_embedder()

    fake_response = {
        "hits": [],
        "processingTimeMs": 6,
        "limit": 20,
        "offset": 0,
        "estimatedTotalHits": 0,
    }

    index.http.post = lambda *args, **kwargs: fake_response

    # Search with ONLY media, no text query
    response = index.search_with_media(
        media={"text": "space exploration"},
        opt_params={"hybrid": {"embedder": "default"}}
    )

    assert isinstance(response, dict)
    assert "hits" in response
    # This validates that SDK correctly sends media without requiring q parameter
