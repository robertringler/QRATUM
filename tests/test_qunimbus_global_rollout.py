"""
Unit tests for QuNimbus Global Rollout components
Tests RL optimizer, pilot generator, and benchmarking functionality
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qunimbus" / "rl"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestMultiVerticalOptimizer:
    """Test RL optimizer for multi-vertical autonomous adaptation"""

    def test_optimizer_initialization(self):
        from multi_vertical_optimizer import MultiVerticalOptimizer

        verticals = ["automotive", "pharma", "energy"]
        optimizer = MultiVerticalOptimizer(verticals, {})

        assert optimizer.verticals == verticals
        assert optimizer.policy_version == "1.0.0"
        assert optimizer.convergence_target == 0.95

    def test_pilot_generation(self):
        from multi_vertical_optimizer import MultiVerticalOptimizer

        verticals = ["automotive", "pharma"]
        optimizer = MultiVerticalOptimizer(verticals, {})

        pilots = optimizer.generate_pilots(5)

        assert len(pilots) == 5
        assert all("id" in p for p in pilots)
        assert all("vertical" in p for p in pilots)
        assert all("timestamp" in p for p in pilots)
        assert all("fidelity" in p for p in pilots)
        assert all(p["fidelity"] == 0.995 for p in pilots)

    def test_policy_adaptation(self):
        from multi_vertical_optimizer import MultiVerticalOptimizer

        optimizer = MultiVerticalOptimizer(["automotive"], {})
        convergence = optimizer.adapt_policy({})

        assert convergence == 0.95


class TestInfinitePilotFactory:
    """Test infinite pilot generation with procedural generation"""

    def test_factory_initialization(self):
        from infinite_pilot_generator import InfinitePilotFactory

        factory = InfinitePilotFactory(10, "automotive,pharma", "rl/opt.py")

        assert factory.rate == 10
        assert factory.verticals == ["automotive", "pharma"]
        assert factory.rl_hook == "rl/opt.py"

    def test_pilot_generation_automotive(self):
        from infinite_pilot_generator import InfinitePilotFactory

        factory = InfinitePilotFactory(1, "automotive", "rl/opt.py")
        pilot = factory.generate_pilot("automotive", 0)

        assert pilot["pilot_id"] == "000-automotive"
        assert pilot["vertical"] == "automotive"
        assert pilot["workload"] == "QPE Battery Chem"
        assert pilot["fidelity"] >= 0.99
        assert pilot["status"] == "active"
        assert "x" in pilot["efficiency_gain"].lower()

    def test_pilot_generation_pharma(self):
        from infinite_pilot_generator import InfinitePilotFactory

        factory = InfinitePilotFactory(1, "pharma", "rl/opt.py")
        pilot = factory.generate_pilot("pharma", 1)

        assert pilot["vertical"] == "pharma"
        assert pilot["workload"] == "VQE Protein Fold"
        assert 0.99 <= pilot["fidelity"] <= 1.0

    def test_batch_generation(self):
        # Use temporary directory for test
        import tempfile

        from infinite_pilot_generator import InfinitePilotFactory

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = InfinitePilotFactory(5, "automotive,pharma", "rl/opt.py")
            factory.output_dir = Path(tmpdir)

            pilots = factory.generate_batch()

            assert len(pilots) == 5
            # Check files were created
            pilot_files = list(Path(tmpdir).glob("*.json"))
            assert len(pilot_files) == 5


class TestMultiVerticalBenchmark:
    """Test benchmarking across AWS/GCP/Azure"""

    def test_benchmark_initialization(self):
        from benchmark_multi_vertical import MultiVerticalBenchmark

        benchmark = MultiVerticalBenchmark("automotive,pharma", "18x")

        assert benchmark.verticals == ["automotive", "pharma"]
        assert benchmark.target == "18x"
        assert benchmark.clouds == ["AWS", "GCP", "Azure"]

    def test_single_vertical_benchmark(self):
        from benchmark_multi_vertical import MultiVerticalBenchmark

        benchmark = MultiVerticalBenchmark("automotive", "18x")
        result = benchmark.benchmark_vertical("automotive")

        assert result["vertical"] == "automotive"
        assert "qunimbus" in result
        assert "aws" in result
        assert result["qunimbus"]["throughput"] > result["aws"]["throughput"]
        assert result["qunimbus"]["fidelity"] >= 0.995
        assert "18" in result["speedup"]

    def test_efficiency_comparison(self):
        from benchmark_multi_vertical import MultiVerticalBenchmark

        benchmark = MultiVerticalBenchmark("pharma", "18x")
        result = benchmark.benchmark_vertical("pharma")

        qn_efficiency = result["qunimbus"]["throughput"] / result["qunimbus"]["cost_per_hour"]
        aws_efficiency = result["aws"]["throughput"] / result["aws"]["cost_per_hour"]

        assert qn_efficiency > aws_efficiency * 10  # At least 10x better


class TestGlobalRolloutIntegration:
    """Integration tests for global rollout task"""

    def test_yaml_task_definition_valid(self):
        import yaml

        task_path = (
            Path(__file__).parent.parent
            / ".github"
            / "copilot-tasks"
            / "qunimbus_global_rollout.yaml"
        )

        with open(task_path) as f:
            task = yaml.safe_load(f)

        assert "name" in task
        assert "steps" in task
        assert len(task["steps"]) == 12
        assert "deliverables" in task
        assert "validation" in task

    def test_workflow_definition_valid(self):
        import yaml

        workflow_path = (
            Path(__file__).parent.parent / ".github" / "workflows" / "qunimbus-global-ci.yml"
        )

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert "jobs" in workflow
        assert "lint" in workflow["jobs"]
        assert "validate" in workflow["jobs"]
        assert "generate_pilots" in workflow["jobs"]

    def test_sdk_structure_automotive(self):
        sdk_path = Path(__file__).parent.parent / "sdk" / "automotive" / "python" / "__init__.py"
        assert sdk_path.exists()

        content = sdk_path.read_text()
        assert "QuNimbusClient" in content
        assert "__version__" in content
        assert "__vertical__" in content

    def test_sdk_structure_pharma(self):
        sdk_path = Path(__file__).parent.parent / "sdk" / "pharma" / "python" / "__init__.py"
        assert sdk_path.exists()

        content = sdk_path.read_text()
        assert "QuNimbusClient" in content
        assert "pharma" in content


def run_tests():
    """Run all tests"""
    print("Running QuNimbus Global Rollout Tests\n")

    # Test RL optimizer
    print("Testing MultiVerticalOptimizer...")
    test_optimizer = TestMultiVerticalOptimizer()
    test_optimizer.test_optimizer_initialization()
    print("  ✓ Optimizer initialization")
    test_optimizer.test_pilot_generation()
    print("  ✓ Pilot generation")
    test_optimizer.test_policy_adaptation()
    print("  ✓ Policy adaptation")

    # Test pilot factory
    print("\nTesting InfinitePilotFactory...")
    test_factory = TestInfinitePilotFactory()
    test_factory.test_factory_initialization()
    print("  ✓ Factory initialization")
    test_factory.test_pilot_generation_automotive()
    print("  ✓ Automotive pilot generation")
    test_factory.test_pilot_generation_pharma()
    print("  ✓ Pharma pilot generation")
    test_factory.test_batch_generation()
    print("  ✓ Batch generation")

    # Test benchmarking
    print("\nTesting MultiVerticalBenchmark...")
    test_bench = TestMultiVerticalBenchmark()
    test_bench.test_benchmark_initialization()
    print("  ✓ Benchmark initialization")
    test_bench.test_single_vertical_benchmark()
    print("  ✓ Single vertical benchmark")
    test_bench.test_efficiency_comparison()
    print("  ✓ Efficiency comparison")

    # Integration tests
    print("\nTesting Global Rollout Integration...")
    test_integration = TestGlobalRolloutIntegration()
    test_integration.test_yaml_task_definition_valid()
    print("  ✓ YAML task definition")
    test_integration.test_workflow_definition_valid()
    print("  ✓ Workflow definition")
    test_integration.test_sdk_structure_automotive()
    print("  ✓ SDK structure (automotive)")
    test_integration.test_sdk_structure_pharma()
    print("  ✓ SDK structure (pharma)")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
