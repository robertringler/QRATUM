"""Rate limiting, audit logging, and request validation middleware.

This module provides middleware for:
- Rate limiting (token bucket algorithm)
- Audit logging (NIST 800-53 AU controls)
- Request validation
- Request ID generation and correlation
- Security headers

Compliance:
- NIST 800-53 AU-2: Audit events
- NIST 800-53 AU-3: Content of audit records
- NIST 800-53 SC-5: Denial of service protection
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

try:
    from fastapi import FastAPI, Request, Response, status
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration.

    Attributes:
        requests_per_minute: Maximum requests per minute per client
        burst_size: Maximum burst size
        enabled: Whether rate limiting is enabled
    """

    requests_per_minute: int = 100
    burst_size: int = 20
    enabled: bool = True


class TokenBucket:
    """Token bucket rate limiter.

    Implements the token bucket algorithm for rate limiting with:
    - Configurable refill rate
    - Burst capacity
    - Per-client tracking
    """

    def __init__(self, rate: float, capacity: int):
        """Initialize token bucket.

        Args:
            rate: Tokens per second refill rate
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}  # client_id -> (tokens, last_update)

    def allow(self, client_id: str) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier (IP, user ID, etc.)

        Returns:
            True if request is allowed
        """
        now = time.time()
        tokens, last_update = self._buckets.get(client_id, (self.capacity, now))

        # Refill tokens based on elapsed time
        elapsed = now - last_update
        tokens = min(self.capacity, tokens + elapsed * self.rate)

        if tokens >= 1:
            self._buckets[client_id] = (tokens - 1, now)
            return True

        self._buckets[client_id] = (tokens, now)
        return False

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until next allowed request.

        Args:
            client_id: Client identifier

        Returns:
            Seconds until next token is available
        """
        tokens, _ = self._buckets.get(client_id, (self.capacity, time.time()))
        if tokens >= 1:
            return 0
        return int((1 - tokens) / self.rate) + 1


# Global rate limiter
_rate_limiter: TokenBucket | None = None


def get_rate_limiter(config: RateLimitConfig | None = None) -> TokenBucket:
    """Get or create rate limiter.

    Args:
        config: Rate limit configuration

    Returns:
        Token bucket rate limiter
    """
    global _rate_limiter
    if _rate_limiter is None:
        config = config or RateLimitConfig()
        _rate_limiter = TokenBucket(
            rate=config.requests_per_minute / 60.0, capacity=config.burst_size
        )
    return _rate_limiter


def configure_rate_limiter(rate: float, capacity: int) -> None:
    """Configure the rate limiter (for testing purposes).

    Args:
        rate: Tokens per second refill rate
        capacity: Maximum bucket capacity
    """
    global _rate_limiter
    _rate_limiter = TokenBucket(rate=rate, capacity=capacity)


if FASTAPI_AVAILABLE:

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Rate limiting middleware using token bucket algorithm."""

        def __init__(self, app: FastAPI, config: RateLimitConfig | None = None):
            """Initialize middleware.

            Args:
                app: FastAPI application
                config: Rate limit configuration
            """
            super().__init__(app)
            self.config = config or RateLimitConfig()
            self.limiter = get_rate_limiter(self.config)

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Process request with rate limiting.

            Args:
                request: Incoming request
                call_next: Next middleware or route handler

            Returns:
                Response
            """
            if not self.config.enabled:
                return await call_next(request)

            # Get client identifier (prefer user ID, fall back to IP)
            client_id = self._get_client_id(request)

            if not self.limiter.allow(client_id):
                retry_after = self.limiter.get_retry_after(client_id)
                logger.warning("Rate limit exceeded for client: %s", client_id)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limit_exceeded",
                        "error_description": "Too many requests",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.config.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    },
                )

            return await call_next(request)

        def _get_client_id(self, request: Request) -> str:
            """Extract client identifier from request.

            Args:
                request: Incoming request

            Returns:
                Client identifier string
            """
            # Check for authenticated user
            if hasattr(request.state, "user") and request.state.user:
                return f"user:{request.state.user.get('user_id', 'unknown')}"

            # Fall back to IP address
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return f"ip:{forwarded.split(',')[0].strip()}"

            client_host = request.client.host if request.client else "unknown"
            return f"ip:{client_host}"

    class AuditLogMiddleware(BaseHTTPMiddleware):
        """Audit logging middleware for NIST 800-53 compliance.

        Logs:
        - Request timestamp
        - Client identifier
        - HTTP method and path
        - Response status
        - Request duration
        - Correlation ID
        """

        def __init__(self, app: FastAPI, excluded_paths: list[str] | None = None):
            """Initialize middleware.

            Args:
                app: FastAPI application
                excluded_paths: Paths to exclude from logging (e.g., /health)
            """
            super().__init__(app)
            self.excluded_paths = excluded_paths or ["/health", "/readiness", "/metrics"]
            self.audit_logger = logging.getLogger("qratum.audit")

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Process request with audit logging.

            Args:
                request: Incoming request
                call_next: Next middleware or route handler

            Returns:
                Response
            """
            # Skip excluded paths
            if request.url.path in self.excluded_paths:
                return await call_next(request)

            # Generate request ID
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            request.state.request_id = request_id

            start_time = time.time()
            timestamp = datetime.utcnow().isoformat() + "Z"

            try:
                response = await call_next(request)
                duration_ms = (time.time() - start_time) * 1000

                # Log successful request
                self._log_request(
                    request_id=request_id,
                    timestamp=timestamp,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    client_ip=self._get_client_ip(request),
                    user_id=getattr(request.state, "user_id", None),
                )

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                return response

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self._log_request(
                    request_id=request_id,
                    timestamp=timestamp,
                    method=request.method,
                    path=request.url.path,
                    status_code=500,
                    duration_ms=duration_ms,
                    client_ip=self._get_client_ip(request),
                    user_id=getattr(request.state, "user_id", None),
                    error=str(e),
                )
                raise

        def _get_client_ip(self, request: Request) -> str:
            """Extract client IP from request."""
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            return request.client.host if request.client else "unknown"

        def _log_request(
            self,
            request_id: str,
            timestamp: str,
            method: str,
            path: str,
            status_code: int,
            duration_ms: float,
            client_ip: str,
            user_id: str | None = None,
            error: str | None = None,
        ) -> None:
            """Log request to audit log.

            Args:
                request_id: Request correlation ID
                timestamp: Request timestamp
                method: HTTP method
                path: Request path
                status_code: Response status code
                duration_ms: Request duration in milliseconds
                client_ip: Client IP address
                user_id: Authenticated user ID
                error: Error message if any
            """
            log_data = {
                "request_id": request_id,
                "timestamp": timestamp,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": client_ip,
                "user_id": user_id or "anonymous",
            }

            if error:
                log_data["error"] = error

            log_level = logging.INFO if status_code < 400 else logging.WARNING
            self.audit_logger.log(
                log_level,
                "AUDIT: method=%s path=%s status=%d duration=%.2fms client=%s user=%s request_id=%s",
                method,
                path,
                status_code,
                duration_ms,
                client_ip,
                user_id or "anonymous",
                request_id,
            )

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        """Middleware to add security headers to responses.

        Headers:
        - X-Content-Type-Options: nosniff
        - X-Frame-Options: DENY
        - X-XSS-Protection: 1; mode=block
        - Strict-Transport-Security: max-age=31536000
        - Cache-Control: no-store for API responses
        """

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Add security headers to response.

            Args:
                request: Incoming request
                call_next: Next middleware

            Returns:
                Response with security headers
            """
            response = await call_next(request)

            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"

            return response

    class RequestValidationMiddleware(BaseHTTPMiddleware):
        """Middleware for request validation.

        Validates:
        - Content-Type header for POST/PUT/PATCH
        - Request body size limits
        - Required headers
        """

        def __init__(self, app: FastAPI, max_body_size: int = 10 * 1024 * 1024):  # 10MB default
            """Initialize middleware.

            Args:
                app: FastAPI application
                max_body_size: Maximum request body size in bytes
            """
            super().__init__(app)
            self.max_body_size = max_body_size

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Validate request.

            Args:
                request: Incoming request
                call_next: Next middleware

            Returns:
                Response
            """
            # Check content length
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_body_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "request_too_large",
                        "error_description": f"Request body exceeds {self.max_body_size} bytes",
                    },
                )

            # Check content type for requests with body
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("Content-Type", "")
                if content_type and not (
                    content_type.startswith("application/json")
                    or content_type.startswith("application/x-www-form-urlencoded")
                    or content_type.startswith("multipart/form-data")
                ):
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "error": "unsupported_media_type",
                            "error_description": "Content-Type must be application/json",
                        },
                    )

            return await call_next(request)


def setup_middleware(app: FastAPI, config: dict[str, Any] | None = None) -> None:
    """Set up all middleware for the application.

    Args:
        app: FastAPI application
        config: Optional middleware configuration
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    config = config or {}

    # Add middleware in reverse order (last added = first executed)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        RequestValidationMiddleware, max_body_size=config.get("max_body_size", 10 * 1024 * 1024)
    )
    app.add_middleware(
        AuditLogMiddleware,
        excluded_paths=config.get("audit_excluded_paths", ["/health", "/readiness"]),
    )
    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(
            requests_per_minute=config.get("rate_limit_rpm", 100),
            burst_size=config.get("rate_limit_burst", 20),
            enabled=config.get("rate_limit_enabled", True),
        ),
    )

    logger.info("Middleware configured successfully")
