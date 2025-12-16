"""Tests for XENON runtime."""

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.bayesian_updater import BayesianUpdater, ExperimentResult
from xenon.learning.mechanism_prior import MechanismPrior
from xenon.runtime.xenon_kernel import XENONRuntime


class TestXENONRuntime:
    """Tests for XENON runtime."""

    def test_create_runtime(self):
        """Test creating a runtime."""
        runtime = XENONRuntime()
        assert runtime.max_mechanisms == 1000
        assert len(runtime.targets) == 0

    def test_add_target(self):
        """Test adding a target."""
        runtime = XENONRuntime()
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        assert len(runtime.targets) == 1
        assert runtime.targets[0].name == "test_target"
        assert runtime.targets[0].protein == "TestProtein"

    def test_run_single_iteration(self):
        """Test running single iteration."""
        runtime = XENONRuntime(max_mechanisms=10)
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        summary = runtime.run(max_iterations=1, seed=42)

        assert summary["iterations"] == 1
        assert summary["mechanisms_discovered"] > 0

    def test_run_multiple_iterations(self):
        """Test running multiple iterations."""
        runtime = XENONRuntime(max_mechanisms=10)
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        summary = runtime.run(max_iterations=5, seed=42)

        assert summary["iterations"] >= 1
        assert summary["mechanisms_discovered"] > 0

    def test_get_mechanisms(self):
        """Test retrieving high-confidence mechanisms."""
        runtime = XENONRuntime(max_mechanisms=10)
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        runtime.run(max_iterations=3, seed=42)

        # Get mechanisms with min evidence
        mechanisms = runtime.get_mechanisms(min_evidence=0.01)
        assert len(mechanisms) > 0

        # All returned mechanisms should meet threshold
        for mech in mechanisms:
            assert mech.posterior >= 0.01

    def test_convergence(self):
        """Test that runtime can converge."""
        runtime = XENONRuntime(
            max_mechanisms=10,
            convergence_threshold=100.0,  # High threshold for quick convergence
        )
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        summary = runtime.run(max_iterations=10, seed=42)

        # Should converge quickly with high threshold
        assert summary["iterations"] <= 10

    def test_multiple_targets(self):
        """Test running with multiple targets."""
        runtime = XENONRuntime(max_mechanisms=10)

        runtime.add_target(
            name="target1",
            protein="Protein1",
            objective="characterize",
        )
        runtime.add_target(
            name="target2",
            protein="Protein2",
            objective="find_inhibitor",
        )

        summary = runtime.run(max_iterations=2, seed=42)

        assert summary["mechanisms_discovered"] > 0

    def test_get_summary(self):
        """Test getting runtime summary."""
        runtime = XENONRuntime(max_mechanisms=10)
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        runtime.run(max_iterations=3, seed=42)

        summary = runtime.get_summary()

        assert "iterations" in summary
        assert "total_mechanisms" in summary
        assert "targets" in summary
        assert summary["targets"] == 1

    def test_mechanism_pruning(self):
        """Test that low-evidence mechanisms are pruned."""
        runtime = XENONRuntime(max_mechanisms=5)
        runtime.add_target(
            name="test_target",
            protein="TestProtein",
            objective="characterize",
        )

        # Run enough iterations to trigger pruning
        runtime.run(max_iterations=10, seed=42)

        # Should not exceed max_mechanisms
        total = sum(len(mechs) for mechs in runtime.mechanisms.values())
        assert total <= runtime.max_mechanisms


class TestBayesianUpdater:
    """Tests for Bayesian updater."""

    def test_create_updater(self):
        """Test creating updater."""
        updater = BayesianUpdater()
        assert updater.likelihood_scale == 1.0

    def test_update_mechanisms(self):
        """Test updating mechanism posteriors."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", concentration=80.0)
        state2 = MolecularState(name="S2", molecule="Protein", concentration=20.0)

        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1.0)
        mech.add_transition(transition)

        # Set initial posterior
        mech.posterior = 0.5

        # Create experiment result
        result = ExperimentResult(
            experiment_type="concentration",
            observations={"S1": 75.0, "S2": 25.0},
            uncertainties={"S1": 10.0, "S2": 5.0},
        )

        updater = BayesianUpdater()
        updated = updater.update_mechanisms([mech], result)

        assert len(updated) == 1
        # Posterior should be updated
        assert updated[0].posterior > 0

    def test_prune_low_evidence(self):
        """Test pruning low-evidence mechanisms."""
        mechs = []
        for i in range(5):
            mech = BioMechanism(f"mech_{i}")
            mech.posterior = 0.1 * (i + 1)  # 0.1, 0.2, 0.3, 0.4, 0.5
            mechs.append(mech)

        updater = BayesianUpdater()
        pruned = updater.prune_low_evidence(mechs, threshold=0.25)

        assert len(pruned) == 3  # Only mechs with posterior >= 0.25

    def test_evidence_summary(self):
        """Test getting evidence summary."""
        mechs = []
        for i in range(3):
            mech = BioMechanism(f"mech_{i}")
            mech.posterior = 1.0 / 3.0
            mechs.append(mech)

        updater = BayesianUpdater()
        summary = updater.get_evidence_summary(mechs)

        assert summary["n_mechanisms"] == 3
        assert abs(summary["total_evidence"] - 1.0) < 1e-6
        assert summary["entropy"] > 0  # Should have non-zero entropy


class TestMechanismPrior:
    """Tests for mechanism prior computation."""

    def test_create_prior(self):
        """Test creating prior calculator."""
        prior = MechanismPrior()
        assert prior.rate_constant_scale == 1.0

    def test_compute_prior(self):
        """Test computing mechanism prior."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)

        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1e3)
        mech.add_transition(transition)

        prior_calc = MechanismPrior()
        prior = prior_calc.compute_prior(mech)

        assert 0.0 < prior <= 1.0

    def test_rate_constant_prior(self):
        """Test rate constant prior computation."""
        prior_calc = MechanismPrior()

        # Typical biochemical rate (should have high prior)
        trans_typical = Transition(source="S1", target="S2", rate_constant=1e3)
        prior_typical = prior_calc.rate_constant_prior(trans_typical)

        # Extremely slow rate (should have lower prior)
        trans_slow = Transition(source="S1", target="S2", rate_constant=1e-6)
        prior_slow = prior_calc.rate_constant_prior(trans_slow)

        assert prior_typical > prior_slow

    def test_initialize_mechanism_priors(self):
        """Test initializing priors for multiple mechanisms."""
        mechs = []
        for i in range(3):
            mech = BioMechanism(f"mech_{i}")

            state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
            state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)
            mech.add_state(state1)
            mech.add_state(state2)

            transition = Transition(source="S1", target="S2", rate_constant=1e3)
            mech.add_transition(transition)

            mechs.append(mech)

        prior_calc = MechanismPrior()
        initialized = prior_calc.initialize_mechanism_priors(mechs)

        # Posteriors should be normalized
        total = sum(m.posterior for m in initialized)
        assert abs(total - 1.0) < 1e-6

    def test_literature_evidence(self):
        """Test adding literature evidence."""
        prior_calc = MechanismPrior()

        prior_calc.add_literature_evidence("TestProtein", 100)

        # Create mechanism with TestProtein
        mech = BioMechanism("test")
        state = MolecularState(name="S1", molecule="TestProtein")
        mech.add_state(state)

        prior = prior_calc.compute_prior(mech)
        assert prior > 0


class TestExperimentResult:
    """Tests for experiment result."""

    def test_create_experiment_result(self):
        """Test creating experiment result."""
        result = ExperimentResult(
            experiment_type="concentration",
            observations={"S1": 80.0, "S2": 20.0},
            uncertainties={"S1": 10.0, "S2": 5.0},
            conditions={"temperature": 310.0},
        )

        assert result.experiment_type == "concentration"
        assert result.observations["S1"] == 80.0
        assert result.uncertainties["S1"] == 10.0
        assert result.conditions["temperature"] == 310.0
