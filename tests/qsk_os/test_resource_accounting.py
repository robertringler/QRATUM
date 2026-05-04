from qsk.resources import ResourceBudget


def test_resource_consumption_and_release():
    budget = ResourceBudget(cpu=4, memory=8)
    assert budget.consume(cpu=2, memory=4)
    assert not budget.consume(cpu=3, memory=0)
    budget.release(cpu=1, memory=2)
    snapshot = budget.snapshot()
    assert snapshot["cpu"] == 1
    assert snapshot["memory"] == 2
