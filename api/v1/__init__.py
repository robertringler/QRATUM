"""QRATUM Platform API v1.

This package provides the FastAPI implementation for the QRATUM platform,
including authentication, job management, status monitoring, and resource allocation.

Modules:
    main: FastAPI application factory
    auth: OAuth2/OIDC authentication with Vault integration
    jobs: Job submission and management endpoints
    status: Real-time status monitoring (WebSocket)
    results: Results retrieval and visualization
    resources: Resource allocation dashboard
    middleware: Rate limiting, audit logging, request validation
"""

from api.v1.main import create_app

__all__ = ["create_app"]
__version__ = "1.0.0"
