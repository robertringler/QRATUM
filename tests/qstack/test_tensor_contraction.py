from qstack.quasim.core.tensor_contraction import contract_tensors


def test_contract_tensors_deterministic():
    result = contract_tensors([[1.0, 2.0], [3.0, 4.0]])
    assert result == [1.0 * 3.0 + 2.0 * 4.0]


def test_contract_shape_mismatch():
    try:
        contract_tensors([[1.0], [1.0, 2.0]])
    except ValueError:
        return
    assert False, "Expected ValueError not raised"
