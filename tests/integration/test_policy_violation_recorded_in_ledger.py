from qledger.query import LedgerQuery
from qledger.record import LedgerRecord
from qledger.store import LedgerStore


def test_policy_violation_recorded_and_queryable():
    store = LedgerStore()
    store.append("main", LedgerRecord.genesis())
    store.append(
        "main",
        LedgerRecord(
            tick=4,
            record_type="violation",
            payload={"article": "safety-envelope", "detail": "missing syscall"},
            node_id="n1",
        ),
    )
    query = LedgerQuery(store)
    violations = query.violations_for_node("n1")
    assert len(violations) == 1
    assert violations[0].payload["article"] == "safety-envelope"
