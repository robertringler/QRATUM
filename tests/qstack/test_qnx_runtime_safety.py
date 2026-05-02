from qstack.qnx import (QNXVM, DeterministicScheduler, OperatorLibrary,
                        QNXState, RateLimiter, SafetyConstraints,
                        SafetyEnvelope, SafetyValidator, TraceRecorder)


def test_scheduler_and_safety_with_rate_limit():
    state = QNXState()
    ops = OperatorLibrary()
    ops.register("alpha", lambda s, g: g.get("alpha", 0) + 1)
    ops.register("beta", lambda s, g: g.get("beta", 0) + 1)

    constraints = SafetyConstraints(rules=[lambda s, g: True])
    envelope = SafetyEnvelope(bounds={"alpha": (0, 5)})
    limiter = RateLimiter(limit_key="cycles", max_calls=2)
    validator = SafetyValidator(constraints, envelope, rate_limiter=limiter)

    vm = QNXVM(DeterministicScheduler(ops), validator, tracer=TraceRecorder())
    result = vm.run_cycle(state, {"alpha": 1, "beta": 2})
    assert "trace" in result
    assert result["trace"][0]["op"] == "alpha"
    assert result["recorded"]

    vm.run_cycle(state, {"alpha": 1, "beta": 2})
    third = vm.run_cycle(state, {"alpha": 1, "beta": 2})
    assert third.get("error") == "safety_pre_check_failed"
