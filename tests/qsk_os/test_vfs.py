from qsk.fs_sim import VirtualFileSystem


def test_vfs_write_read_and_fingerprint():
    vfs = VirtualFileSystem()
    vfs.mkdir("/data")
    vfs.write("/data/file.txt", "hello")
    assert vfs.read("/data/file.txt") == "hello"
    assert vfs.list("/data") == ["file.txt"]
    assert "file:file.txt:5" in vfs.fingerprint()
