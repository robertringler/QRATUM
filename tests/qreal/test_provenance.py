from qreal.provenance import compute_provenance


def test_provenance_is_deterministic():
    payload = {"a": 1, "b": 2}
    first = compute_provenance("source", payload, 5)
    second = compute_provenance("source", payload, 5)
    assert first == second
    assert first.digest == second.digest
