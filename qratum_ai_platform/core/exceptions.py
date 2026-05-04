"""QRATUM Platform Exception Types.

Custom exceptions for platform integration layer.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations


class QRATUMPlatformError(Exception):
    """Base exception for QRATUM platform errors."""

    pass


class PlatformConfigError(QRATUMPlatformError):
    """Raised when platform configuration is invalid."""

    pass


class ComplianceValidationError(QRATUMPlatformError):
    """Raised when DO-178C compliance checks fail."""

    pass


class BackendSelectionError(QRATUMPlatformError):
    """Raised when backend selection fails."""

    pass


class WorkflowExecutionError(QRATUMPlatformError):
    """Raised when workflow execution fails."""

    pass


__all__ = [
    "QRATUMPlatformError",
    "PlatformConfigError",
    "ComplianceValidationError",
    "BackendSelectionError",
    "WorkflowExecutionError",
]
