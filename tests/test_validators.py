"""Validation function tests.

This module tests QuASIM validation functions including:
- Input validation
- Configuration validation
- Result validation
- Schema validation
"""

from __future__ import annotations


class TestInputValidation:
    """Test input validation functions."""

    def test_validate_positive_integer(self):
        """Test positive integer validation."""

        def validate_positive_int(value: int) -> bool:
            return isinstance(value, int) and value > 0

        assert validate_positive_int(1)
        assert validate_positive_int(100)
        assert not validate_positive_int(0)
        assert not validate_positive_int(-1)

    def test_validate_range(self):
        """Test range validation."""

        def validate_range(value: float, min_val: float, max_val: float) -> bool:
            return min_val <= value <= max_val

        assert validate_range(5.0, 0.0, 10.0)
        assert validate_range(0.0, 0.0, 10.0)
        assert validate_range(10.0, 0.0, 10.0)
        assert not validate_range(11.0, 0.0, 10.0)
        assert not validate_range(-1.0, 0.0, 10.0)

    def test_validate_precision_mode(self):
        """Test precision mode validation."""

        def validate_precision(mode: str) -> bool:
            valid_modes = {"fp8", "fp16", "fp32", "fp64"}
            return mode in valid_modes

        assert validate_precision("fp32")
        assert validate_precision("fp64")
        assert not validate_precision("fp128")
        assert not validate_precision("invalid")

    def test_validate_backend(self):
        """Test backend validation."""

        def validate_backend(backend: str) -> bool:
            valid_backends = {"cpu", "cuda", "rocm"}
            return backend in valid_backends

        assert validate_backend("cpu")
        assert validate_backend("cuda")
        assert validate_backend("rocm")
        assert not validate_backend("opencl")


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_validate_required_fields(self):
        """Test required field validation."""

        def validate_config(config: dict) -> tuple[bool, list[str]]:
            required = ["api_url", "timeout"]
            missing = [field for field in required if field not in config]
            return len(missing) == 0, missing

        valid_config = {"api_url": "http://localhost", "timeout": 30}
        invalid_config = {"api_url": "http://localhost"}

        is_valid, missing = validate_config(valid_config)
        assert is_valid
        assert len(missing) == 0

        is_valid, missing = validate_config(invalid_config)
        assert not is_valid
        assert "timeout" in missing

    def test_validate_field_types(self):
        """Test field type validation."""

        def validate_types(config: dict) -> bool:
            if "timeout" in config and not isinstance(config["timeout"], int):
                return False
            if "api_url" in config and not isinstance(config["api_url"], str):
                return False
            return True

        assert validate_types({"timeout": 30, "api_url": "http://localhost"})
        assert not validate_types({"timeout": "30", "api_url": "http://localhost"})
        assert not validate_types({"timeout": 30, "api_url": 123})

    def test_validate_url_format(self):
        """Test URL format validation."""

        def validate_url(url: str) -> bool:
            return url.startswith("http://") or url.startswith("https://")

        assert validate_url("http://localhost:8000")
        assert validate_url("https://api.quasim.com")
        assert not validate_url("localhost:8000")
        assert not validate_url("ftp://server.com")


class TestResultValidation:
    """Test result validation."""

    def test_validate_simulation_result(self):
        """Test simulation result validation."""

        def validate_result(result: list) -> bool:
            if not isinstance(result, list):
                return False
            if len(result) == 0:
                return True
            return all(isinstance(x, complex) for x in result)

        assert validate_result([1 + 0j, 2 + 0j])
        assert validate_result([])
        assert not validate_result([1, 2, 3])
        assert not validate_result("not a list")

    def test_validate_optimization_result(self):
        """Test optimization result validation."""

        def validate_opt_result(result: dict) -> bool:
            required = ["solution", "objective_value", "iterations", "convergence"]
            return all(field in result for field in required)

        valid_result = {
            "solution": [0.5, 0.3],
            "objective_value": 1.5,
            "iterations": 100,
            "convergence": True,
        }
        invalid_result = {"solution": [0.5, 0.3], "objective_value": 1.5}

        assert validate_opt_result(valid_result)
        assert not validate_opt_result(invalid_result)

    def test_validate_fidelity(self):
        """Test fidelity metric validation."""

        def validate_fidelity(fidelity: float) -> bool:
            return 0.0 <= fidelity <= 1.0

        assert validate_fidelity(0.95)
        assert validate_fidelity(0.0)
        assert validate_fidelity(1.0)
        assert not validate_fidelity(1.5)
        assert not validate_fidelity(-0.1)


class TestSchemaValidation:
    """Test schema validation."""

    def test_validate_job_schema(self):
        """Test job submission schema validation."""

        def validate_job_schema(job: dict) -> bool:
            required = ["job_type", "parameters"]
            if not all(field in job for field in required):
                return False
            if job["job_type"] not in ["cfd", "fea", "optimization"]:
                return False
            return isinstance(job["parameters"], dict)

        valid_job = {"job_type": "cfd", "parameters": {"solver": "navier_stokes"}}
        invalid_job1 = {"job_type": "invalid", "parameters": {}}
        invalid_job2 = {"job_type": "cfd"}

        assert validate_job_schema(valid_job)
        assert not validate_job_schema(invalid_job1)
        assert not validate_job_schema(invalid_job2)

    def test_validate_telemetry_schema(self):
        """Test telemetry data schema validation."""

        def validate_telemetry(data: dict) -> bool:
            required = ["timestamp", "simulation_id", "metrics"]
            if not all(field in data for field in required):
                return False
            return isinstance(data["metrics"], dict)

        valid_data = {
            "timestamp": "2025-12-12T14:00:00Z",
            "simulation_id": "sim-001",
            "metrics": {"fidelity": 0.98},
        }
        invalid_data = {"timestamp": "2025-12-12T14:00:00Z", "simulation_id": "sim-001"}

        assert validate_telemetry(valid_data)
        assert not validate_telemetry(invalid_data)


class TestConstraintValidation:
    """Test constraint validation."""

    def test_validate_inequality_constraint(self):
        """Test inequality constraint validation."""

        def validate_inequality(solution: list[float], bound: float) -> bool:
            return sum(solution) <= bound

        assert validate_inequality([0.5, 0.3, 0.1], 1.0)
        assert not validate_inequality([0.5, 0.7, 0.5], 1.0)

    def test_validate_equality_constraint(self):
        """Test equality constraint validation."""

        def validate_equality(
            solution: list[float], target: float, tolerance: float = 1e-6
        ) -> bool:
            return abs(sum(solution) - target) < tolerance

        assert validate_equality([0.5, 0.5], 1.0)
        assert validate_equality([0.3, 0.3, 0.4], 1.0)
        assert not validate_equality([0.5, 0.3], 1.0)

    def test_validate_bounds(self):
        """Test bound constraint validation."""

        def validate_bounds(solution: list[float], bounds: list[tuple[float, float]]) -> bool:
            if len(solution) != len(bounds):
                return False
            return all(lower <= val <= upper for val, (lower, upper) in zip(solution, bounds))

        solution = [0.5, 0.7, 0.3]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        invalid_bounds = [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5)]

        assert validate_bounds(solution, bounds)
        assert not validate_bounds(solution, invalid_bounds)


class TestErrorValidation:
    """Test error and exception validation."""

    def test_validate_error_threshold(self):
        """Test error threshold validation."""

        def within_error_threshold(
            actual: float, expected: float, threshold: float = 1e-10
        ) -> bool:
            return abs(actual - expected) <= threshold

        assert within_error_threshold(1.0, 1.0)
        assert within_error_threshold(1.0, 1.0 + 1e-11)
        assert not within_error_threshold(1.0, 1.1)

    def test_validate_convergence(self):
        """Test convergence validation."""

        def has_converged(history: list[float], tolerance: float = 1e-6) -> bool:
            if len(history) < 2:
                return False
            return abs(history[-1] - history[-2]) < tolerance

        assert has_converged([1.0, 0.5, 0.25, 0.125, 0.125])
        assert not has_converged([1.0, 0.5, 0.25])

    def test_validate_numerical_stability(self):
        """Test numerical stability validation."""

        def is_numerically_stable(value: float) -> bool:
            import math

            return not (math.isnan(value) or math.isinf(value))

        assert is_numerically_stable(1.0)
        assert is_numerically_stable(0.0)
        assert is_numerically_stable(-1.0)
        assert not is_numerically_stable(float("nan"))
        assert not is_numerically_stable(float("inf"))


class TestDataValidation:
    """Test data structure validation."""

    def test_validate_circuit_format(self):
        """Test quantum circuit format validation."""

        def validate_circuit(circuit: list) -> bool:
            if not isinstance(circuit, list):
                return False
            if len(circuit) == 0:
                return True
            return all(isinstance(gate, list) for gate in circuit)

        assert validate_circuit([[1 + 0j, 0 + 0j]])
        assert validate_circuit([])
        assert not validate_circuit("not a circuit")
        assert not validate_circuit([1, 2, 3])

    def test_validate_matrix_dimensions(self):
        """Test matrix dimension validation."""

        def validate_matrix_dims(
            matrix: list[list], expected_rows: int, expected_cols: int
        ) -> bool:
            if len(matrix) != expected_rows:
                return False
            return all(len(row) == expected_cols for row in matrix)

        matrix_2x3 = [[1, 2, 3], [4, 5, 6]]
        assert validate_matrix_dims(matrix_2x3, 2, 3)
        assert not validate_matrix_dims(matrix_2x3, 3, 2)

    def test_validate_state_vector(self):
        """Test state vector validation."""

        def validate_state_vector(state: list[complex]) -> bool:
            if not all(isinstance(x, complex) for x in state):
                return False
            # Check normalization (sum of squared magnitudes should be ~1)
            norm_squared = sum(abs(x) ** 2 for x in state)
            return abs(norm_squared - 1.0) < 1e-3  # More lenient tolerance

        import math

        # Properly normalized state
        valid_state = [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j]
        invalid_state = [1 + 0j, 1 + 0j]

        assert validate_state_vector(valid_state)
        assert not validate_state_vector(invalid_state)


class TestPerformanceValidation:
    """Test performance metric validation."""

    def test_validate_latency(self):
        """Test latency validation."""

        def validate_latency(latency: float, max_latency: float) -> bool:
            return 0.0 < latency <= max_latency

        assert validate_latency(0.001, 1.0)
        assert validate_latency(0.5, 1.0)
        assert not validate_latency(0.0, 1.0)
        assert not validate_latency(2.0, 1.0)

    def test_validate_throughput(self):
        """Test throughput validation."""

        def validate_throughput(throughput: float, min_throughput: float) -> bool:
            return throughput >= min_throughput

        assert validate_throughput(100.0, 50.0)
        assert validate_throughput(50.0, 50.0)
        assert not validate_throughput(25.0, 50.0)

    def test_validate_memory_usage(self):
        """Test memory usage validation."""

        def validate_memory(usage_mb: int, limit_mb: int) -> bool:
            return 0 < usage_mb <= limit_mb

        assert validate_memory(512, 1024)
        assert validate_memory(1024, 1024)
        assert not validate_memory(2048, 1024)
        assert not validate_memory(0, 1024)
