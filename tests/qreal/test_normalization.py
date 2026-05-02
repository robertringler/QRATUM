from qreal.normalizers import NormalizationChain, clamp_numbers, sort_keys


def test_normalization_chain_applies_steps_in_order():
    chain = NormalizationChain([clamp_numbers(0, 10), sort_keys])
    payload = {"b": 20, "a": -5}
    normalized = chain.apply(payload)
    assert list(normalized.keys()) == ["a", "b"]
    assert normalized["a"] == 0
    assert normalized["b"] == 10
