from qunimbus.synthetic.credit_network import CreditNetwork


def test_contagion_losses_are_deterministic():
    network = CreditNetwork()
    network.add_exposure("l1", "b1", 5.0)
    network.add_exposure("l2", "b1", 3.0)
    network.add_exposure("l1", "b2", 2.0)

    losses = network.contagion("b1")

    assert losses == {"l1": 5.0, "l2": 3.0}
