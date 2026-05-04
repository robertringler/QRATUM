from qstack.q import Attestor, IdentityRegistry, KeyManager, Ledger, QIdentity, Signer, TrustGraph


def test_identity_attestation_and_registry():
    km = KeyManager(seed="seed")
    key = km.derive_key("alice")
    signer = Signer(key)
    attestor = Attestor(signer)
    ledger = Ledger()
    registry = IdentityRegistry(attestor=attestor, ledger=ledger)

    alice = QIdentity(name="alice", key=key)
    record = registry.register(alice, {"role": "operator"})

    assert registry.verify("alice")
    assert ledger.verify()
    assert record.ledger_index == 0


def test_trust_graph_path():
    km = KeyManager(seed="seed")
    a = QIdentity("a", km.derive_key("a"))
    b = QIdentity("b", km.derive_key("b"))
    c = QIdentity("c", km.derive_key("c"))

    graph = TrustGraph()
    graph.add_trust(a, b)
    graph.add_trust(b, c)

    assert graph.verify_path(a, c)
    attestor = Attestor(Signer(km.derive_key("att")))
    attested = graph.attest_edge(attestor, a, b)
    assert attestor.verify(attested)
