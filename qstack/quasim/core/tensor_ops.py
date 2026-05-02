"""Deterministic tensor operations using pure Python lists."""

from __future__ import annotations


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if len(a[0]) != len(b):
        raise ValueError("shapes do not align for matmul")
    result: list[list[float]] = []
    for i in range(len(a)):
        row: list[float] = []
        for j in range(len(b[0])):
            row.append(sum(a[i][k] * b[k][j] for k in range(len(b))))
        result.append(row)
    return result


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector lengths do not match")
    return float(sum(x * y for x, y in zip(a, b)))


def tensor_contract(tensor: list[list[float]]) -> float:
    return float(sum(sum(row) for row in tensor))


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def hadamard(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("shapes do not align for hadamard product")
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def kron(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for row_a in a:
        for row_b in b:
            result.append([elem_a * elem_b for elem_a in row_a for elem_b in row_b])
    return result


def gate_apply(state: list[float], gate: list[list[float]]) -> list[float]:
    if len(gate) != len(state):
        raise ValueError("gate dimension must match state length")
    return [sum(gate[i][j] * state[j] for j in range(len(state))) for i in range(len(gate))]


def normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]
