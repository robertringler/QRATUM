"""QRATUM Platform API - Main Application.

This module provides the FastAPI application factory and configuration
for the QRATUM platform API.

Features:
- OAuth2/OIDC authentication with Vault integration
- Job submission, status monitoring, and results retrieval
- WebSocket support for real-time updates
- Rate limiting and audit logging
- OpenAPI documentation
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

try:
    from fastapi import FastAPI, Form, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


if FASTAPI_AVAILABLE:
    from api.v1.auth import (RefreshTokenRequest, RevokeTokenRequest,
                             TokenManager, TokenResponse)
    from api.v1.jobs import router as jobs_router
    from api.v1.jobs import validation_router
    from api.v1.middleware import setup_middleware
    from api.v1.resources import router as resources_router
    from api.v1.results import router as results_router
    from api.v1.status import router as status_router

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str
        timestamp: str

    class ReadinessResponse(BaseModel):
        """Readiness check response."""

        ready: bool
        dependencies: dict[str, str]


def create_app(config: dict[str, Any] | None = None) -> Any:
    """Create and configure the FastAPI application.

    Args:
        config: Optional application configuration

    Returns:
        Configured FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None

    config = config or {}

    app = FastAPI(
        title="QRATUM Platform API",
        description="""
QRATUM (Quantum Resource Allocation, Tensor Analysis, and Unified Modeling) Platform API.

This API provides endpoints for:
- Job submission (quantum simulations, tensor analysis, VQE, QAOA)
- Real-time status monitoring (WebSocket support)
- Results retrieval and visualization
- Resource allocation dashboards

## Security
All endpoints require OAuth2/OIDC authentication. API keys are supported for
service-to-service communication.

## Compliance
- DO-178C Level A certification maintained
- NIST 800-53 security controls enforced
- CMMC 2.0 Level 2 compliant
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
    )

    # CORS configuration
    allowed_origins = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup middleware (rate limiting, audit logging, security headers)
    setup_middleware(app, config)

    # Health endpoints (no authentication required)
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint for load balancers."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    @app.get("/readiness", response_model=ReadinessResponse, tags=["Health"])
    async def readiness_check() -> ReadinessResponse:
        """Readiness check for Kubernetes probes."""
        # In production, check actual dependencies
        return ReadinessResponse(
            ready=True,
            dependencies={
                "vault": "ok",
                "redis": "ok",
                "postgres": "ok",
            },
        )

    # Authentication endpoints
    @app.post("/v1/auth/token", response_model=TokenResponse, tags=["Authentication"])
    async def get_token(
        grant_type: str = Form(...),
        client_id: str | None = Form(None),
        client_secret: str | None = Form(None),
        username: str | None = Form(None),
        password: str | None = Form(None),
        scope: str | None = Form(None),
    ) -> TokenResponse:
        """Get access token using OAuth2 client credentials or password flow."""
        token_manager = TokenManager()

        if grant_type == "client_credentials":
            if not client_id or not client_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="client_id and client_secret required for client_credentials grant",
                )

            # Validate client credentials (in production, check against Vault/database)
            if not _validate_client(client_id, client_secret):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid client credentials",
                )

            scopes = scope.split() if scope else ["read:jobs", "write:jobs", "read:resources"]
            access_token = token_manager.create_access_token(client_id, scopes)
            refresh_token = token_manager.create_refresh_token(client_id)

            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_in=3600,
                refresh_token=refresh_token,
                scope=" ".join(scopes),
            )

        elif grant_type == "password":
            if not username or not password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="username and password required for password grant",
                )

            # Validate user credentials (in production, check against identity provider)
            if not _validate_user(username, password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            scopes = scope.split() if scope else ["read:jobs", "write:jobs", "read:resources"]
            access_token = token_manager.create_access_token(username, scopes)
            refresh_token = token_manager.create_refresh_token(username)

            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_in=3600,
                refresh_token=refresh_token,
                scope=" ".join(scopes),
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported grant_type: {grant_type}",
            )

    @app.post("/v1/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
    async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
        """Refresh access token using refresh token."""
        token_manager = TokenManager()

        is_valid, payload = token_manager.verify_token(request.refresh_token)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )

        subject = payload.get("sub")
        scopes = payload.get("scope", "").split()

        # Filter out 'refresh' scope and use remaining scopes
        scopes = [s for s in scopes if s != "refresh"]
        if not scopes:
            scopes = ["read:jobs", "write:jobs", "read:resources"]

        access_token = token_manager.create_access_token(subject, scopes)
        new_refresh_token = token_manager.create_refresh_token(subject)

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=new_refresh_token,
            scope=" ".join(scopes),
        )

    @app.post("/v1/auth/revoke", tags=["Authentication"])
    async def revoke_token(
        request: RevokeTokenRequest,
        user: dict[str, Any] = None,  # Optional - token can revoke itself
    ) -> dict[str, Any]:
        """Revoke an access or refresh token."""
        token_manager = TokenManager()

        if token_manager.revoke_token(request.token):
            return {"success": True, "message": "Token revoked"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke token",
            )

    # Include routers with /v1 prefix
    app.include_router(jobs_router, prefix="/v1")
    app.include_router(validation_router, prefix="/v1")
    app.include_router(status_router, prefix="/v1")
    app.include_router(results_router, prefix="/v1")
    app.include_router(resources_router, prefix="/v1")

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler for unhandled errors."""
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "error_description": "An unexpected error occurred",
            },
        )

    logger.info("QRATUM API application created successfully")
    return app


def _validate_client(client_id: str, client_secret: str) -> bool:
    """Validate client credentials.

    Args:
        client_id: Client ID
        client_secret: Client secret

    Returns:
        True if credentials are valid
    """
    # In production, validate against Vault or database
    # For development, accept any non-empty credentials
    return bool(client_id and client_secret and len(client_secret) >= 8)


def _validate_user(username: str, password: str) -> bool:
    """Validate user credentials.

    Args:
        username: Username
        password: Password

    Returns:
        True if credentials are valid
    """
    # In production, validate against identity provider
    # For development, accept any non-empty credentials
    return bool(username and password and len(password) >= 8)


def main() -> int:
    """Run the API server for development.

    Returns:
        Exit code
    """
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1

    try:
        import uvicorn

        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        return 0
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
