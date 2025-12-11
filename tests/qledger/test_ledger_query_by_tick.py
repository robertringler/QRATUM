from qledger.query import LedgerQuery
from qledger.record import LedgerRecord
from qledger.store import LedgerStore


def test_query_by_tick_and_constitution():
    store = LedgerStore()
    store.append("main", LedgerRecord.genesis())
    store.append(
        "main",
        LedgerRecord(tick=5, record_type="constitution_version", payload={"version": "v1"}, node_id="sys"),
    )
    store.append(
        "main",
        LedgerRecord(tick=7, record_type="constitution_version", payload={"version": "v2"}, node_id="sys"),
    )

    query = LedgerQuery(store)
    records_tick5 = query.records_between(0, 5)
    assert any(rec.payload.get("version") == "v1" for rec in records_tick5)
    active = query.active_constitution_at(6)
    assert active.payload["version"] == "v1"
    active_latest = query.active_constitution_at(8)
    assert active_latest.payload["version"] == "v2"
