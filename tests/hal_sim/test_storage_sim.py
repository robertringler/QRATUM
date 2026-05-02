from qsk.hal.storage import BlockDevice


def test_block_device_read_write():
    dev = BlockDevice(blocks=2, block_size=4)
    payload = b"abcd"
    dev.write(0, payload)
    assert dev.read(0) == payload
    assert dev.checksum() > 0
