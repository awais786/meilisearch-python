"""Tests for client experimental features methods."""


def test_get_experimental_features(client):
    """Test getting experimental features returns dict with multimodal feature."""
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

    # Reset
    client.update_experimental_features({"multimodal": initial_multimodal})


def test_enable_disable_multimodal(client):
    """Test enable and disable multimodal convenience methods."""
    # Test enable
    response = client.enable_multimodal()
    assert response.get("multimodal") is True
    assert client.get_experimental_features()["multimodal"] is True

    # Test disable
    response = client.disable_multimodal()
    assert response.get("multimodal") is False
    assert client.get_experimental_features()["multimodal"] is False


def test_multimodal_idempotency(client):
    """Test that enable/disable operations are idempotent."""
    # Enable twice - should not error
    client.enable_multimodal()
    response = client.enable_multimodal()
    assert response.get("multimodal") is True

    # Disable twice - should not error
    client.disable_multimodal()
    response = client.disable_multimodal()
    assert response.get("multimodal") is False



