"""Deterministic tensor operations using pure Python lists."""

from __future__ import annotations

from typing import List


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if len(a[0]) != len(b):
        raise ValueError("shapes do not align for matmul")
    result: List[List[float]] = []
    for i in range(len(a)):
        row: List[float] = []
        for j in range(len(b[0])):
            row.append(sum(a[i][k] * b[k][j] for k in range(len(b))))
        result.append(row)
    return result


def dot(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector lengths do not match")
    return float(sum(x * y for x, y in zip(a, b)))


def tensor_contract(tensor: List[List[float]]) -> float:
    return float(sum(sum(row) for row in tensor))


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix)]


def hadamard(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("shapes do not align for hadamard product")
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def kron(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result: List[List[float]] = []
    for row_a in a:
        for row_b in b:
            result.append([elem_a * elem_b for elem_a in row_a for elem_b in row_b])
    return result


def gate_apply(state: List[float], gate: List[List[float]]) -> List[float]:
    if len(gate) != len(state):
        raise ValueError("gate dimension must match state length")
    return [sum(gate[i][j] * state[j] for j in range(len(state))) for i in range(len(gate))]


def normalize(vec: List[float]) -> List[float]:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]
