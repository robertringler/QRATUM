from qledger.chain import LedgerChain
from qledger.record import LedgerRecord


def test_chain_append_and_hash_links():
    chain = LedgerChain()
    first = chain.append(LedgerRecord.genesis())
    second = chain.append(LedgerRecord(tick=1, record_type="event", payload={"k": 1}, node_id="n1"))
    third = chain.append(LedgerRecord(tick=2, record_type="event", payload={"k": 2}, node_id="n1"))

    assert first.prev_hash == "GENESIS"
    assert second.prev_hash == first.compute_hash()
    assert third.prev_hash == second.compute_hash()
    assert chain.validate()
