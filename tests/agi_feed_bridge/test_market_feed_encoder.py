from qnx_agi.perception.market_feed_encoder import encode


def test_market_encoder_wraps_payload():
    payload = {"symbol": "Q", "close": 10}
    percept = encode(payload)
    assert percept.kind == "market"
    assert percept.payload == payload
