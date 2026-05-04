"""Tests for authentication endpoints."""

import pytest


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_readiness_check(self, api_client):
        """Test readiness check endpoint."""
        response = api_client.get("/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "dependencies" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_get_token_client_credentials(self, api_client):
        """Test getting token with client credentials."""
        response = api_client.post(
            "/v1/auth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret-key-12345",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "Bearer"
        assert "expires_in" in data
        assert "refresh_token" in data

    def test_get_token_password_grant(self, api_client):
        """Test getting token with password grant."""
        response = api_client.post(
            "/v1/auth/token",
            data={
                "grant_type": "password",
                "username": "testuser",
                "password": "testpassword123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_get_token_missing_credentials(self, api_client):
        """Test token request with missing credentials."""
        response = api_client.post(
            "/v1/auth/token",
            data={
                "grant_type": "client_credentials",
            },
        )
        assert response.status_code == 400

    def test_get_token_invalid_credentials(self, api_client):
        """Test token request with invalid credentials."""
        response = api_client.post(
            "/v1/auth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "invalid",
                "client_secret": "short",
            },
        )
        assert response.status_code == 401

    def test_refresh_token(self, api_client, auth_headers):
        """Test token refresh."""
        # First get a token
        response = api_client.post(
            "/v1/auth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret-key-12345",
            },
        )
        refresh_token = response.json().get("refresh_token")
        if not refresh_token:
            pytest.skip("Refresh token not available")

        # Refresh the token
        response = api_client.post(
            "/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_refresh_token_invalid(self, api_client):
        """Test token refresh with invalid token."""
        response = api_client.post(
            "/v1/auth/refresh",
            json={"refresh_token": "invalid-token"},
        )
        assert response.status_code == 401
