from qsk.hal.net import NetworkLink, Packet, route
from quasim.domains.network_sim import simulate_latency


def test_network_routing_and_latency():
    net = {("a", "b"): NetworkLink(latency=2)}
    pkt = Packet(source="a", destination="b", payload="data")
    delivered = route(net, pkt)
    assert delivered == []
    delivered = route(net, pkt)
    assert delivered[0].payload == "data"
    assert simulate_latency({("a", "b"): 2}, ("a", "b")) == 4
