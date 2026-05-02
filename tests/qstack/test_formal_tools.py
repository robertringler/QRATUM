from qstack.qnx_agi.formal import (AbstractLattice, AbstractState, Interval,
                                   ModelChecker, SymbolicExecutor,
                                   TLASpecification, propagate_affine)


def test_interval_arithmetic_operations():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    assert a.add(b).to_tuple() == (4.0, 6.0)
    assert a.sub(b).to_tuple() == (-3.0, -1.0)
    assert a.mul(b).to_tuple() == (3.0, 8.0)


def test_interval_affine_propagation_is_deterministic():
    env = {
        "a": Interval(1.0, 2.0),
        "b": Interval(-1.0, 0.0),
    }
    weights = {"a": 2.0, "b": 0.5}
    result = propagate_affine(env, weights, bias=1.0)
    # Expected bounds: bias=1.0, a contributes [2.0, 4.0], b contributes [-0.5, 0.0]
    assert result.to_tuple() == (2.5, 5.0)
    # repeated calls produce identical bounds when inputs are unchanged
    repeat = propagate_affine(env, weights, bias=1.0)
    assert repeat.to_tuple() == result.to_tuple()


def test_symbolic_execution_and_model_checking():
    executor = SymbolicExecutor(lambda state: {"x": state["x"] + 1.0})
    intervals = executor.execute({"x": Interval(0.0, 0.0)}, steps=3)
    assert intervals["x"].contains(3.0)

    spec = TLASpecification(
        name="counter",
        init=lambda s: s.get("x", 0) == 0,
        next_state=lambda s: {"x": s.get("x", 0) + 1},
        invariant=lambda s: s.get("x", 0) >= 0,
    )
    checker = ModelChecker(specification=spec, bound=3)
    states = checker.run({"x": 0})
    assert states[-1]["x"] == 3


def test_abstract_state_join_and_widen():
    lattice = AbstractLattice()
    s1 = AbstractState(lattice)
    s1.assign("x", Interval(0.0, 1.0))
    s2 = AbstractState(lattice)
    s2.assign("x", Interval(2.0, 3.0))
    joined = s1.join(s2)
    assert joined.read("x").to_tuple() == (0.0, 3.0)

    widened = s2.widen(s1)
    assert widened.read("x").low in {0.0, float("-inf")}
