from qstack.q import (
    Attestor,
    CapabilityAuthority,
    DeterministicAccessControlList,
    DeterministicKeyExchange,
    DeterministicLedger,
    DeterministicRevocationList,
    KeyManager,
    QIdentity,
    Signer,
    SovereignClusterReplication,
)


def test_merkle_and_ledger_replication():
    km = KeyManager("seed")
    alice = QIdentity("alice", km.derive_key("alice"))
    attestor = Attestor(signer=Signer(km.derive_key("attestor")))
    ledger_a = DeterministicLedger(attestor=attestor)
    ledger_b = DeterministicLedger(attestor=attestor)
    ledger_a.append({"owner": alice.name})
    ledger_a.append({"payload": "data"})
    cluster = SovereignClusterReplication()
    cluster.register_node("a", ledger_a)
    cluster.register_node("b", ledger_b)
    divergence = cluster.detect_divergence()
    assert divergence == ["b"]
    cluster.reconcile("a")
    assert ledger_b.verify()
    assert cluster.detect_divergence() == []


def test_caps_and_acl():
    km = KeyManager("seed")
    alice = QIdentity("alice", km.derive_key("alice"))
    bob = QIdentity("bob", km.derive_key("bob"))
    acl = DeterministicAccessControlList()
    acl.grant(alice, "read")
    assert acl.allowed(alice, "read")
    authority = CapabilityAuthority()
    token = authority.issue(alice, bob, "write")
    assert authority.validate(token, bob, "write", alice)


def test_revocation_and_kex():
    km = KeyManager("seed")
    alice = QIdentity("alice", km.derive_key("alice"))
    bob = QIdentity("bob", km.derive_key("bob"))
    kex = DeterministicKeyExchange()
    shared1 = kex.derive_shared(alice, bob)
    shared2 = kex.derive_shared(bob, alice)
    assert shared1 == shared2
    revocations = DeterministicRevocationList()
    revocations.revoke(alice, "rotated")
    assert revocations.is_revoked(alice)
    assert revocations.verify_reason(alice, "rotated")
