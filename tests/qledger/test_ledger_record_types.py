from qledger.record import LedgerRecord
from qledger.store import LedgerStore


def test_record_type_classification():
    store = LedgerStore()
    store.append("main", LedgerRecord.genesis())
    store.append(
        "main",
        LedgerRecord(tick=2, record_type="violation", payload={"article": "safety-envelope"}, node_id="n1"),
    )
    store.append(
        "main",
        LedgerRecord(tick=3, record_type="policy_change", payload={"policy_id": "p1"}, node_id="n1"),
    )

    violations = store.by_type("violation")
    assert len(violations) == 1
    assert violations[0].payload["article"] == "safety-envelope"
    policies = store.by_type("policy_change")
    assert policies[0].payload["policy_id"] == "p1"
