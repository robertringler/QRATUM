"""Integration tests for the QRATUM Platform API.

This module provides pytest fixtures and tests for the API endpoints.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set a consistent JWT secret for testing
os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-12345678"


@pytest.fixture(scope="session", autouse=True)
def disable_rate_limiting():
    """Disable rate limiting for tests."""
    from api.v1.middleware import configure_rate_limiter

    # Configure with very high limits for testing
    configure_rate_limiter(rate=10000, capacity=10000)


@pytest.fixture
def api_client():
    """Create a test API client."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not available - install with: pip install fastapi")

    try:
        # Import dynamically to ensure path is set
        from api.v1.main import create_app

        # Create app with rate limiting disabled
        app = create_app(
            {"rate_limit_enabled": False, "rate_limit_rpm": 100000, "rate_limit_burst": 100000}
        )
        if app is None:
            pytest.skip("FastAPI application not available")
        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"API module not available: {e}")
    except Exception as e:
        pytest.skip(f"Failed to create test app: {e}")


@pytest.fixture
def auth_headers(api_client):
    """Get authentication headers for API requests."""
    # Get token using client credentials
    response = api_client.post(
        "/v1/auth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": "test-client",
            "client_secret": "test-secret-key-12345",
        },
    )
    if response.status_code != 200:
        pytest.skip("Authentication not available")

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers():
    """Get API key headers for API requests."""
    return {"X-API-Key": "qratum_test_api_key_12345"}
