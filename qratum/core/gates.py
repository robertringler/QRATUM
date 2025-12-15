"""QRATUM quantum gate library.

Complete set of quantum gates including single-qubit gates (Pauli, Hadamard,
phase gates, rotation gates) and multi-qubit gates (CNOT, CZ, SWAP, Toffoli).
"""

import numpy as np
from typing import Union

# Type alias for gate matrices
GateMatrix = np.ndarray


# ============================================================================
# Single-Qubit Gates
# ============================================================================

# Pauli Gates
I = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity
X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X (NOT)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z

# Hadamard Gate
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase Gates
S = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase gate (sqrt(Z))
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T gate (sqrt(S))
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)  # S-dagger
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)  # T-dagger


def RX(theta: float) -> GateMatrix:
    """Rotation around X-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def RY(theta: float) -> GateMatrix:
    """Rotation around Y-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def RZ(theta: float) -> GateMatrix:
    """Rotation around Z-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def Phase(phi: float) -> GateMatrix:
    """Phase gate with arbitrary angle.

    Args:
        phi: Phase angle in radians

    Returns:
        2x2 phase matrix
    """
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


def U3(theta: float, phi: float, lam: float) -> GateMatrix:
    """General single-qubit rotation gate (U3).

    Args:
        theta: Rotation angle around Y-axis
        phi: Rotation angle for final Z-rotation
        lam: Rotation angle for initial Z-rotation

    Returns:
        2x2 general rotation matrix
    """
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )


# ============================================================================
# Two-Qubit Gates
# ============================================================================

# CNOT gate (Control-NOT)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)

# CZ gate (Control-Z)
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)

# SWAP gate
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)

# iSWAP gate
iSWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def CRX(theta: float) -> GateMatrix:
    """Controlled rotation around X-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        4x4 controlled rotation matrix
    """
    rx = RX(theta)
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), rx]])


def CRY(theta: float) -> GateMatrix:
    """Controlled rotation around Y-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        4x4 controlled rotation matrix
    """
    ry = RY(theta)
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), ry]])


def CRZ(theta: float) -> GateMatrix:
    """Controlled rotation around Z-axis.

    Args:
        theta: Rotation angle in radians

    Returns:
        4x4 controlled rotation matrix
    """
    rz = RZ(theta)
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), rz]])


# ============================================================================
# Three-Qubit Gates
# ============================================================================

# Toffoli gate (CCNOT, CCX)
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[6:8, 6:8] = X

# Fredkin gate (CSWAP)
FREDKIN = np.eye(8, dtype=complex)
FREDKIN[5:7, 5:7] = SWAP[:2, :2]


__all__ = [
    "GateMatrix",
    # Single-qubit gates
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "Sdg",
    "Tdg",
    "RX",
    "RY",
    "RZ",
    "Phase",
    "U3",
    # Two-qubit gates
    "CNOT",
    "CZ",
    "SWAP",
    "iSWAP",
    "CRX",
    "CRY",
    "CRZ",
    # Three-qubit gates
    "TOFFOLI",
    "FREDKIN",
]
