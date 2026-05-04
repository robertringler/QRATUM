from qsk.processes import DeterministicProcess


def test_process_run_order():
    dag = [
        {"id": "b", "epoch": 2},
        {"id": "a", "epoch": 1},
        {"id": "c", "epoch": 2},
    ]
    proc = DeterministicProcess(pid="p1", dag=dag)
    trace, verification = proc.run()
    assert trace == ["p1:a", "p1:b", "p1:c"]
    assert verification == ["verified"]
