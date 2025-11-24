"""Tests for client experimental features methods."""

def test_get_experimental_features(client):
    """Test getting experimental features returns a dict including 'multimodal'."""
    response = client.get_experimental_features()

    assert isinstance(response, dict)
    assert len(response) > 0
    assert "multimodal" in response
    assert isinstance(response["multimodal"], bool)


def test_update_experimental_features(client):
    """Test updating experimental features and verify changes persist."""
    initial = client.get_experimental_features()
    initial_multimodal = initial.get("multimodal", False)

    # Toggle multimodal
    new_value = not initial_multimodal
    response = client.update_experimental_features({"multimodal": new_value})

    assert isinstance(response, dict)
    assert response.get("multimodal") == new_value
    assert client.get_experimental_features().get("multimodal") == new_value

    # Reset to original value
    client.update_experimental_features({"multimodal": initial_multimodal})


def test_multimodal_idempotency_generic(client):
    """Test that updating multimodal via generic method is idempotent."""
    # Enable twice
    client.update_experimental_features({"multimodal": True})
    response = client.update_experimental_features({"multimodal": True})
    assert response.get("multimodal") is True

    # Disable twice
    client.update_experimental_features({"multimodal": False})
    response = client.update_experimental_features({"multimodal": False})
    assert response.get("multimodal") is False
