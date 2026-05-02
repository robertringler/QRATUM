from qsk.ipc import Mailbox


def test_mailbox_send_receive():
    mb = Mailbox()
    mb.send("control", "p1", {"cmd": "start"})
    mb.send("control", "p2", {"cmd": "stop"})
    sender1, msg1 = mb.recv("control")
    sender2, msg2 = mb.recv("control")
    assert sender1 == "p1"
    assert msg2["cmd"] == "stop"
    assert mb.stats()["control"] == 0
