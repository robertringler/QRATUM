from qsk.kernel import Kernel


def test_kernel_orders_and_runs():
    k = Kernel()
    k.register_syscall("echo", lambda x: x)
    dag = [
        {"id": "a", "type": "syscall", "name": "echo", "args": ["hello"], "epoch": 1},
        {"id": "b", "type": "value", "value": 5, "epoch": 0},
    ]
    results = k.run(dag)
    assert results[0] == 5
    assert results[1] == "hello"
    assert k.verify_trace()
