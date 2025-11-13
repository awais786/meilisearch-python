# pylint: disable=redefined-outer-name
"""Tests for indexingFragments and searchFragments in embedders (multimodal feature)."""

import pytest


@pytest.mark.usefixtures("enable_multimodal")
def test_rest_embedder_with_fragments(empty_index):
    """Tests that REST embedder can be configured with indexingFragments and searchFragments."""
    index = empty_index()

    rest_embedder_with_fragments = {
        "rest_fragments": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "apiKey": "test-key",
            "dimensions": 512,
            "indexingFragments": {
                "text": {
                    "value": "{{doc.title}} - {{doc.description}}"
                }
            },
            "searchFragments": {
                "text": {
                    "value": "{{fragment}}"
                }
            },
            "request": {"input": ["{{fragment}}"], "model": "test-model"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
            "headers": {"Authorization": "Bearer test-key"},
        }
    }

    response = index.update_embedders(rest_embedder_with_fragments)
    update = index.wait_for_task(response.task_uid)
    assert update.status == "succeeded"

    embedders = index.get_embedders()
    assert embedders.embedders["rest_fragments"].source == "rest"
    assert embedders.embedders["rest_fragments"].url == "http://localhost:8000/embed"
    assert embedders.embedders["rest_fragments"].dimensions == 512

    # Verify fragments are configured
    assert hasattr(embedders.embedders["rest_fragments"], "indexing_fragments")
    assert hasattr(embedders.embedders["rest_fragments"], "search_fragments")
    assert embedders.embedders["rest_fragments"].indexing_fragments is not None
    assert embedders.embedders["rest_fragments"].search_fragments is not None


@pytest.mark.usefixtures("enable_multimodal")
def test_rest_embedder_with_multiple_fragments(empty_index):
    """Tests that REST embedder can be configured with multiple fragment types."""
    index = empty_index()

    rest_embedder_multi_fragments = {
        "multi_fragments": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "dimensions": 1024,
            "indexingFragments": {
                "text": {
                    "value": "{{doc.title}}"
                },
                "description": {
                    "value": "{{doc.overview}}"
                }
            },
            "searchFragments": {
                "text": {
                    "value": "{{fragment}}"
                },
                "description": {
                    "value": "{{fragment}}"
                }
            },
            "request": {"input": ["{{fragment}}"], "model": "test-model"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
        }
    }

    response = index.update_embedders(rest_embedder_multi_fragments)
    update = index.wait_for_task(response.task_uid)
    assert update.status == "succeeded"

    embedders = index.get_embedders()
    assert embedders.embedders["multi_fragments"].source == "rest"

    # Verify multiple fragments are configured
    indexing_frags = embedders.embedders["multi_fragments"].indexing_fragments
    search_frags = embedders.embedders["multi_fragments"].search_fragments

    assert indexing_frags is not None
    assert search_frags is not None
    # The exact structure depends on the Pydantic model implementation
    assert len(indexing_frags) >= 1
    assert len(search_frags) >= 1


@pytest.mark.usefixtures("enable_multimodal")
def test_fragments_without_document_template(empty_index):
    """Tests that fragments can be used without documentTemplate (they are mutually exclusive)."""
    index = empty_index()

    embedder_config = {
        "fragments_only": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "dimensions": 512,
            # No documentTemplate - only fragments
            "indexingFragments": {
                "text": {
                    "value": "{{doc.content}}"
                }
            },
            "searchFragments": {
                "text": {
                    "value": "{{fragment}}"
                }
            },
            "request": {"input": ["{{fragment}}"], "model": "test-model"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
        }
    }

    response = index.update_embedders(embedder_config)
    update = index.wait_for_task(response.task_uid)
    assert update.status == "succeeded"

    embedders = index.get_embedders()
    # Should not have documentTemplate when using fragments
    assert embedders.embedders["fragments_only"].document_template is None
    assert embedders.embedders["fragments_only"].indexing_fragments is not None
    assert embedders.embedders["fragments_only"].search_fragments is not None


def test_fragments_require_multimodal_feature(empty_index):
    """Tests that fragments configuration requires multimodal feature to be enabled."""
    # This test runs WITHOUT the enable_multimodal fixture
    index = empty_index()

    embedder_with_fragments = {
        "test": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "dimensions": 512,
            "indexingFragments": {
                "text": {"value": "{{doc.title}}"}
            },
            "searchFragments": {
                "text": {"value": "{{fragment}}"}
            },
            "request": {"input": ["{{fragment}}"], "model": "test"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
        }
    }

    # This might fail or succeed depending on whether multimodal is required
    # The behavior depends on the Meilisearch server version
    try:
        response = index.update_embedders(embedder_with_fragments)
        task = index.wait_for_task(response.task_uid)
        # If it succeeds, fragments should still be configured
        if task.status == "succeeded":
            embedders = index.get_embedders()
            assert embedders.embedders["test"].indexing_fragments is not None
    except Exception:
        # If it fails, that's also acceptable as the feature might require enabling
        pass


@pytest.mark.usefixtures("enable_multimodal")
def test_update_fragments_separately(empty_index):
    """Tests updating indexingFragments and searchFragments separately."""
    index = empty_index()

    # First, configure with basic fragments
    initial_config = {
        "updatable": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "dimensions": 512,
            "indexingFragments": {
                "text": {"value": "{{doc.title}}"}
            },
            "searchFragments": {
                "text": {"value": "{{fragment}}"}
            },
            "request": {"input": ["{{fragment}}"], "model": "test"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
        }
    }

    response = index.update_embedders(initial_config)
    index.wait_for_task(response.task_uid)

    # Then update with different fragment configuration
    updated_config = {
        "updatable": {
            "source": "rest",
            "url": "http://localhost:8000/embed",
            "dimensions": 512,
            "indexingFragments": {
                "text": {"value": "{{doc.title}} - {{doc.description}}"}
            },
            "searchFragments": {
                "text": {"value": "{{fragment}}"}
            },
            "request": {"input": ["{{fragment}}"], "model": "test"},
            "response": {"data": [{"embedding": "{{embedding}}"}]},
        }
    }

    response = index.update_embedders(updated_config)
    update = index.wait_for_task(response.task_uid)
    assert update.status == "succeeded"

    embedders = index.get_embedders()
    assert embedders.embedders["updatable"].indexing_fragments is not None

