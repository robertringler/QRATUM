from qdl.compiler import Compiler


def test_compiler_lower_executes():
    comp = Compiler("1 + 2")
    fn = comp.lower()
    assert fn() == 3
