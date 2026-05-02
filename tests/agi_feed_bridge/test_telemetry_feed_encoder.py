from qnx_agi.perception.telemetry_feed_encoder import encode


def test_telemetry_encoder_wraps_payload():
    payload = {"vehicle": "sat-1"}
    percept = encode(payload)
    assert percept.kind == "telemetry"
    assert percept.payload == payload
