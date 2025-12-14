"""Tests for Flask application endpoints."""

import json

import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_kernel_endpoint(client):
    """Test the kernel endpoint."""
    response = client.post(
        "/kernel",
        data=json.dumps({"seed": 0, "scale": 1.0}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "result" in data


def test_history_endpoint_empty(client):
    """Test the history endpoint when empty."""
    # Clear any existing history first
    from app import transaction_history

    transaction_history.clear()

    response = client.get("/history")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "transactions" in data
    assert len(data["transactions"]) == 0


def test_history_endpoint_with_transactions(client):
    """Test the history endpoint after creating transactions."""
    from app import transaction_history

    transaction_history.clear()

    # Create a transaction by calling the kernel endpoint
    client.post(
        "/kernel",
        data=json.dumps({"seed": 0, "scale": 1.0}),
        content_type="application/json",
    )

    # Check history
    response = client.get("/history")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "transactions" in data
    assert len(data["transactions"]) == 1
    assert data["transactions"][0]["seed"] == 0
    assert data["transactions"][0]["scale"] == 1.0
    assert "timestamp" in data["transactions"][0]
    assert "result" in data["transactions"][0]


def test_history_limit(client):
    """Test that history is limited to 50 transactions."""
    from app import transaction_history

    transaction_history.clear()

    # Create 60 transactions
    for i in range(60):
        client.post(
            "/kernel",
            data=json.dumps({"seed": i, "scale": 1.0}),
            content_type="application/json",
        )

    # Check that only 50 are kept
    response = client.get("/history")
    data = json.loads(response.data)
    assert len(data["transactions"]) == 50
    # The oldest 10 should have been removed
    assert data["transactions"][0]["seed"] == 10
    assert data["transactions"][-1]["seed"] == 59
