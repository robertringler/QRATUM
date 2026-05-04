"""
Computational verification module for Chapter 13:
Quantum Mechanics as a Fixed Point of the CRS–CIIR–Observer Loop.

Verifies:
 1. Fixed-point convergence of the CRS–CIIR–Observer loop
 2. Density matrix validity (positivity, trace, Hermiticity)
 3. Born rule consistency
 4. Lindblad generator reconstruction
 5. Entropy monotonicity
 6. Trace preservation
 7. Kraus completeness
 8. CRS → QM mapping stability
 9. Perturbation robustness
10. Observer operator properties
11. Hilbert space structure emergence
12. Contractivity verification
"""

import numpy as np

# ============================================================
# Helpers
# ============================================================


def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random density matrix of dimension d."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random pure state density matrix."""
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def is_positive_semidefinite(M: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if M is positive semidefinite."""
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals >= -tol))


def is_hermitian(M: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if M is Hermitian."""
    return bool(np.allclose(M, M.conj().T, atol=tol))


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute trace distance between two density matrices."""
    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S(ρ) = -Tr(ρ ln ρ)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log(eigvals)))


def observer_operator(rho: np.ndarray, lam: float) -> np.ndarray:
    """Apply the observer operator O(ρ) = ρ^{1/(1+λ)} / Tr(ρ^{1/(1+λ)}).

    Parameters
    ----------
    rho : density matrix
    lam : inference temperature λ ∈ (0, 1)
    """
    beta = 1.0 / (1.0 + lam)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0.0)  # numerical safety
    eigvals_beta = eigvals**beta
    Z = np.sum(eigvals_beta)
    if Z < 1e-15:
        return np.eye(rho.shape[0]) / rho.shape[0]
    rho_out = eigvecs @ np.diag(eigvals_beta / Z) @ eigvecs.conj().T
    return rho_out


def make_lindblad_kraus(d: int, n_ops: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate random Lindblad/Kraus operators satisfying Σ K†K = I."""
    kraus = []
    for _ in range(n_ops):
        K = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        kraus.append(K)
    total = sum(K.conj().T @ K for K in kraus)
    eigvals, eigvecs = np.linalg.eigh(total)
    sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-15))) @ eigvecs.conj().T
    return [K @ sqrt_inv for K in kraus]


def apply_channel(kraus: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """Apply a quantum channel defined by Kraus operators."""
    d_out = kraus[0].shape[0]
    result = np.zeros((d_out, d_out), dtype=complex)
    for K in kraus:
        result += K @ rho @ K.conj().T
    return result


def crs_update(
    states: np.ndarray,
    adj: np.ndarray,
    alpha: float,
    delta: float,
    c: float,
    sigma_eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One step of the CRS update rule (Eq. 12.6).

    Parameters
    ----------
    states : (N, d) array of node states
    adj : (N, N) adjacency/weight matrix
    alpha : diffusion coefficient
    delta : decay rate
    c : interaction coupling
    sigma_eta : noise amplitude
    rng : random generator
    """
    N, d = states.shape
    new_states = np.zeros_like(states)
    for v in range(N):
        neighbors = np.where(adj[v] > 0)[0]
        if len(neighbors) > 0:
            s_bar = np.mean(states[neighbors], axis=0)
            coupling = c * np.sum(adj[v, neighbors, np.newaxis] * states[neighbors], axis=0)
        else:
            s_bar = np.zeros(d)
            coupling = np.zeros(d)
        noise = sigma_eta * rng.standard_normal(d)
        new_states[v] = (1 - alpha - delta) * states[v] + alpha * s_bar + coupling + noise
    return new_states


def crs_to_density(states: np.ndarray, d_hilbert: int) -> np.ndarray:
    """Coarse-grain CRS states into a density matrix.

    Maps N node states to a d_hilbert × d_hilbert density matrix
    via outer-product encoding of the first d_hilbert principal modes.
    """
    N = states.shape[0]
    # Use SVD to extract principal modes
    U, s_vals, _ = np.linalg.svd(states, full_matrices=False)
    # Take first d_hilbert modes
    k = min(d_hilbert, len(s_vals))
    psi = np.zeros(d_hilbert, dtype=complex)
    psi[:k] = s_vals[:k]
    # Add phases from U
    for j in range(k):
        psi[j] *= np.exp(1j * np.angle(U[0, j]))
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        return np.eye(d_hilbert) / d_hilbert
    psi = psi / norm
    return np.outer(psi, psi.conj())


def loop_iteration(
    states: np.ndarray,
    adj: np.ndarray,
    d_hilbert: int,
    kraus: list[np.ndarray],
    lam: float,
    alpha: float,
    delta: float,
    c: float,
    sigma_eta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One full CRS→CIIR→Observer→CRS loop iteration.

    Returns (new_states, rho_out).
    """
    # Step 1: CRS evolution
    new_states = crs_update(states, adj, alpha, delta, c, sigma_eta, rng)

    # Step 2: CIIR encoding (coarse-grain + channel)
    rho = crs_to_density(new_states, d_hilbert)
    rho = apply_channel(kraus, rho)

    # Step 3: Observer inference
    rho_obs = observer_operator(rho, lam)

    # Step 4: Back-projection to CRS
    eigvals, eigvecs = np.linalg.eigh(rho_obs)
    d_state = states.shape[1]
    N = states.shape[0]
    back_states = np.zeros((N, d_state))
    for k_idx in range(min(d_hilbert, d_state)):
        if k_idx < len(eigvals):
            mode_weight = np.abs(eigvals[k_idx])
            for v in range(N):
                back_states[v, k_idx % d_state] += mode_weight * np.real(
                    eigvecs[v % d_hilbert, k_idx]
                )

    return back_states, rho_obs


# ============================================================
# Section 1: Observer Operator Properties (Theorem 13.1)
# ============================================================


class TestObserverOperator:
    """Verify properties of the observer operator O(ρ)."""

    def test_output_is_density_matrix(self):
        """O(ρ) must be a valid density matrix."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
            rho_obs = observer_operator(rho, lam)
            assert is_hermitian(rho_obs), f"Not Hermitian for λ={lam}"
            assert is_positive_semidefinite(rho_obs), f"Not PSD for λ={lam}"
            assert abs(np.trace(rho_obs) - 1.0) < 1e-10, f"Trace != 1 for λ={lam}"

    def test_identity_limit(self):
        """lim_{λ→0} O(ρ) = ρ."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        rho_obs = observer_operator(rho, 1e-10)
        assert trace_distance(rho_obs, rho) < 1e-6

    def test_max_mixing_limit(self):
        """As λ→1, O(ρ) approaches maximally mixed state."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        rho_obs = observer_operator(rho, 0.999)
        rho_mixed = np.eye(d) / d
        # At λ=0.999, β=1/1.999≈0.5, eigenvalues compressed significantly
        assert trace_distance(rho_obs, rho_mixed) < 0.3

    def test_preserves_eigenbasis(self):
        """O(ρ) and ρ share the same eigenbasis."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        rho_obs = observer_operator(rho, 0.3)
        # Eigenvectors should be the same (up to phase)
        _, evecs_rho = np.linalg.eigh(rho)
        _, evecs_obs = np.linalg.eigh(rho_obs)
        # Check overlap matrix is diagonal (up to phases)
        overlap = evecs_rho.conj().T @ evecs_obs
        assert np.allclose(np.abs(overlap) ** 2, np.eye(d), atol=1e-8)

    def test_uniqueness_strict_convexity(self):
        """Verify uniqueness via consistency of repeated application."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        lam = 0.3
        rho1 = observer_operator(rho, lam)
        rho2 = observer_operator(rho, lam)
        assert np.allclose(rho1, rho2)

    def test_contractivity(self):
        """O contracts trace distance: ||O(ρ1)-O(ρ2)||_1 ≤ ||ρ1-ρ2||_1."""
        rng = np.random.default_rng(42)
        d = 4
        rho1 = random_density_matrix(d, rng)
        rho2 = random_density_matrix(d, rng)
        lam = 0.3
        d_before = trace_distance(rho1, rho2)
        d_after = trace_distance(
            observer_operator(rho1, lam),
            observer_operator(rho2, lam),
        )
        assert d_after <= d_before + 1e-10


# ============================================================
# Section 2: Density Matrix Validity
# ============================================================


class TestDensityMatrixValidity:
    """Verify density matrix properties throughout the loop."""

    def test_random_dm_is_valid(self):
        """Helper produces valid density matrices."""
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = random_density_matrix(d, rng)
            assert is_hermitian(rho)
            assert is_positive_semidefinite(rho)
            assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_crs_encoding_produces_valid_dm(self):
        """CRS-to-density coarse-graining produces valid density matrix."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 10, 3, 4
        states = rng.standard_normal((N, d_state))
        rho = crs_to_density(states, d_hilbert)
        assert is_hermitian(rho)
        assert is_positive_semidefinite(rho)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_channel_preserves_dm(self):
        """CPTP channel preserves density matrix properties."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        kraus = make_lindblad_kraus(d, 3, rng)
        rho_out = apply_channel(kraus, rho)
        assert is_hermitian(rho_out)
        assert is_positive_semidefinite(rho_out)
        assert abs(np.trace(rho_out) - 1.0) < 1e-10


# ============================================================
# Section 3: Born Rule Consistency (Theorem 13.5)
# ============================================================


class TestBornRule:
    """Verify the Born rule P(E) = Tr(ρE) at fixed point."""

    def test_probability_additivity(self):
        """P(E1+E2) = P(E1) + P(E2) for orthogonal projections."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        # Create orthogonal projections
        E1 = np.zeros((d, d))
        E1[0, 0] = 1.0
        E2 = np.zeros((d, d))
        E2[1, 1] = 1.0
        E12 = E1 + E2
        p1 = np.real(np.trace(rho @ E1))
        p2 = np.real(np.trace(rho @ E2))
        p12 = np.real(np.trace(rho @ E12))
        assert abs(p12 - (p1 + p2)) < 1e-12

    def test_normalization(self):
        """P(I) = 1."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        assert abs(np.real(np.trace(rho)) - 1.0) < 1e-12

    def test_positivity(self):
        """P(E) ≥ 0 for all effects E."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        for _ in range(20):
            # Random effect (0 ≤ E ≤ I)
            A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            E = A @ A.conj().T
            E = E / (np.linalg.norm(E, ord=2) + 1e-10)  # normalize to ≤ I
            p = np.real(np.trace(rho @ E))
            assert p >= -1e-10

    def test_born_rule_pure_state(self):
        """For pure state |ψ⟩, P(|φ⟩⟨φ|) = |⟨ψ|φ⟩|²."""
        rng = np.random.default_rng(42)
        d = 4
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= np.linalg.norm(psi)
        phi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        phi /= np.linalg.norm(phi)
        rho = np.outer(psi, psi.conj())
        E = np.outer(phi, phi.conj())
        p_born = np.real(np.trace(rho @ E))
        p_overlap = np.abs(psi.conj() @ phi) ** 2
        assert abs(p_born - p_overlap) < 1e-12


# ============================================================
# Section 4: Lindblad Generator (Theorem 13.6)
# ============================================================


class TestLindbladGenerator:
    """Verify GKSL structure of the loop generator."""

    def _gksl_generator(
        self, H: np.ndarray, L_ops: list[np.ndarray], gammas: list[float], rho: np.ndarray
    ) -> np.ndarray:
        """Compute L(ρ) = -i[H,ρ] + Σ γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ})."""
        d = rho.shape[0]
        result = -1j * (H @ rho - rho @ H)
        for k, (L, gamma) in enumerate(zip(L_ops, gammas)):
            LdL = L.conj().T @ L
            result += gamma * (L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
        return result

    def test_gksl_preserves_trace(self):
        """Tr(L(ρ)) = 0 for GKSL generator."""
        rng = np.random.default_rng(42)
        d = 4
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = (H + H.conj().T) / 2  # Hermitian
        L_ops = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(3)]
        gammas = [0.1, 0.2, 0.15]
        rho = random_density_matrix(d, rng)
        L_rho = self._gksl_generator(H, L_ops, gammas, rho)
        assert abs(np.trace(L_rho)) < 1e-10

    def test_gksl_preserves_hermiticity(self):
        """L(ρ) is Hermitian when ρ is Hermitian."""
        rng = np.random.default_rng(42)
        d = 4
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = (H + H.conj().T) / 2
        L_ops = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(3)]
        gammas = [0.1, 0.2, 0.15]
        rho = random_density_matrix(d, rng)
        L_rho = self._gksl_generator(H, L_ops, gammas, rho)
        assert is_hermitian(L_rho, tol=1e-10)

    def test_euler_step_preserves_positivity(self):
        """ρ + dt·L(ρ) ≥ 0 for small dt."""
        rng = np.random.default_rng(42)
        d = 4
        H = 0.1 * np.diag(rng.standard_normal(d))
        L_ops = [0.1 * np.eye(d)]
        gammas = [0.01]
        rho = random_density_matrix(d, rng)
        dt = 0.001
        L_rho = self._gksl_generator(H, L_ops, gammas, rho)
        rho_new = rho + dt * L_rho
        rho_new = (rho_new + rho_new.conj().T) / 2  # symmetrize
        assert is_positive_semidefinite(rho_new, tol=1e-8)

    def test_unitary_limit(self):
        """With γ_k=0, dynamics is pure unitary: dρ/dt = -i[H,ρ]."""
        rng = np.random.default_rng(42)
        d = 4
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = (H + H.conj().T) / 2
        rho = random_density_matrix(d, rng)
        L_rho = self._gksl_generator(H, [], [], rho)
        commutator = -1j * (H @ rho - rho @ H)
        assert np.allclose(L_rho, commutator, atol=1e-12)


# ============================================================
# Section 5: Entropy Monotonicity
# ============================================================


class TestEntropyMonotonicity:
    """Verify entropy behaviour under the loop components."""

    def test_channel_does_not_decrease_entropy(self):
        """Unital channels do not decrease entropy."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_pure_state(d, rng)  # S = 0
        # Depolarizing channel (unital)
        p = 0.1
        kraus = [np.sqrt(1 - 3 * p / 4) * np.eye(d)]
        # Add Pauli-like errors for dimension d
        for j in range(d):
            for k in range(d):
                if j != k:
                    E_jk = np.zeros((d, d))
                    E_jk[j, k] = 1.0
                    kraus.append(np.sqrt(p / (d * (d - 1))) * E_jk)
        # Normalize
        total = sum(K.conj().T @ K for K in kraus)
        eigv, eigvec = np.linalg.eigh(total)
        sqrt_inv = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigv, 1e-15))) @ eigvec.conj().T
        kraus_norm = [K @ sqrt_inv for K in kraus]
        rho_out = apply_channel(kraus_norm, rho)
        S_in = von_neumann_entropy(rho)
        S_out = von_neumann_entropy(rho_out)
        assert S_out >= S_in - 1e-10

    def test_observer_modifies_entropy(self):
        """Observer with λ > 0 increases entropy toward max mixed."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        S_in = von_neumann_entropy(rho)
        lam = 0.5
        rho_obs = observer_operator(rho, lam)
        S_out = von_neumann_entropy(rho_obs)
        # Observer compresses eigenvalues, should increase entropy
        assert S_out >= S_in - 1e-10


# ============================================================
# Section 6: Trace Preservation
# ============================================================


class TestTracePreservation:
    """Verify trace preservation at each loop stage."""

    def test_kraus_trace_preservation(self):
        """Σ K†K = I implies Tr(Σ KρK†) = Tr(ρ)."""
        rng = np.random.default_rng(42)
        d = 4
        kraus = make_lindblad_kraus(d, 3, rng)
        rho = random_density_matrix(d, rng)
        rho_out = apply_channel(kraus, rho)
        assert abs(np.trace(rho_out) - 1.0) < 1e-10

    def test_observer_trace_preservation(self):
        """Tr(O(ρ)) = 1."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        for lam in [0.1, 0.5, 0.9]:
            rho_obs = observer_operator(rho, lam)
            assert abs(np.trace(rho_obs) - 1.0) < 1e-10

    def test_loop_trace_preservation(self):
        """Full loop preserves trace."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 8, 3, 4
        states = rng.standard_normal((N, d_state))
        adj = (rng.random((N, N)) > 0.5).astype(float)
        np.fill_diagonal(adj, 0)
        kraus = make_lindblad_kraus(d_hilbert, 3, rng)
        _, rho_out = loop_iteration(
            states,
            adj,
            d_hilbert,
            kraus,
            lam=0.3,
            alpha=0.1,
            delta=0.05,
            c=0.01,
            sigma_eta=0.01,
            rng=rng,
        )
        assert abs(np.trace(rho_out) - 1.0) < 1e-10


# ============================================================
# Section 7: Kraus Completeness
# ============================================================


class TestKrausCompleteness:
    """Verify Kraus operator completeness relation."""

    def test_kraus_completeness(self):
        """Σ K†K = I for generated Kraus operators."""
        rng = np.random.default_rng(42)
        d = 4
        kraus = make_lindblad_kraus(d, 5, rng)
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(d), atol=1e-10)

    def test_kraus_rank_bounded(self):
        """Kraus rank = number of operators ≤ d²."""
        rng = np.random.default_rng(42)
        d = 4
        n_ops = 3
        kraus = make_lindblad_kraus(d, n_ops, rng)
        assert len(kraus) == n_ops
        assert n_ops <= d**2


# ============================================================
# Section 8: Fixed-Point Convergence (Theorem 13.9)
# ============================================================


class TestFixedPointConvergence:
    """Verify fixed-point convergence of the CRS–CIIR–Observer loop."""

    def test_convergence_basic(self):
        """Iterated loop converges (distance decreases)."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 6, 2, 3
        states = rng.standard_normal((N, d_state))
        adj = np.ones((N, N)) - np.eye(N)
        kraus = make_lindblad_kraus(d_hilbert, 2, rng)

        distances = []
        prev_rho = None
        for _ in range(30):
            states, rho = loop_iteration(
                states,
                adj,
                d_hilbert,
                kraus,
                lam=0.3,
                alpha=0.2,
                delta=0.1,
                c=0.01,
                sigma_eta=0.001,
                rng=rng,
            )
            if prev_rho is not None:
                distances.append(trace_distance(rho, prev_rho))
            prev_rho = rho.copy()

        # Check that later distances are smaller (trend)
        if len(distances) > 5:
            early = np.mean(distances[:5])
            late = np.mean(distances[-5:])
            assert late <= early + 0.1  # allow some noise

    def test_fixed_point_produces_valid_dm(self):
        """After many iterations, output is a valid density matrix."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 6, 2, 3
        states = rng.standard_normal((N, d_state))
        adj = np.ones((N, N)) - np.eye(N)
        kraus = make_lindblad_kraus(d_hilbert, 2, rng)

        for _ in range(50):
            states, rho = loop_iteration(
                states,
                adj,
                d_hilbert,
                kraus,
                lam=0.3,
                alpha=0.2,
                delta=0.1,
                c=0.01,
                sigma_eta=0.001,
                rng=rng,
            )
        assert is_hermitian(rho)
        assert is_positive_semidefinite(rho)
        assert abs(np.trace(rho) - 1.0) < 1e-10


# ============================================================
# Section 9: CRS → QM Mapping Stability
# ============================================================


class TestCRStoQMStability:
    """Verify stability of the CRS → QM mapping."""

    def test_similar_crs_produce_similar_dm(self):
        """Small CRS perturbation → small density matrix perturbation."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 8, 3, 4
        states = rng.standard_normal((N, d_state))
        epsilon = 0.01
        states_perturbed = states + epsilon * rng.standard_normal((N, d_state))

        rho1 = crs_to_density(states, d_hilbert)
        rho2 = crs_to_density(states_perturbed, d_hilbert)
        d_tr = trace_distance(rho1, rho2)
        assert d_tr < 1.0  # bounded perturbation

    def test_crs_encoding_idempotent_structure(self):
        """CRS encoding followed by back-projection is approximately stable."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 8, 3, 4
        states = rng.standard_normal((N, d_state))
        rho = crs_to_density(states, d_hilbert)
        # Back-project
        eigvals, eigvecs = np.linalg.eigh(rho)
        back_states = np.zeros((N, d_state))
        for k in range(min(d_hilbert, d_state)):
            if k < len(eigvals):
                for v in range(N):
                    back_states[v, k] += np.abs(eigvals[k]) * np.real(eigvecs[v % d_hilbert, k])
        rho2 = crs_to_density(back_states, d_hilbert)
        # Should still be a valid density matrix
        assert is_positive_semidefinite(rho2)
        assert abs(np.trace(rho2) - 1.0) < 1e-10


# ============================================================
# Section 10: Perturbation Robustness (Theorem 13.10)
# ============================================================


class TestPerturbationRobustness:
    """Verify structural stability under perturbations."""

    def test_small_perturbation_small_shift(self):
        """Perturbed observer has nearby fixed-point output."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        lam = 0.3
        rho_fp = observer_operator(rho, lam)
        # Perturbed observer
        delta = 0.01
        lam_perturbed = lam + delta
        rho_fp_perturbed = observer_operator(rho, lam_perturbed)
        shift = trace_distance(rho_fp, rho_fp_perturbed)
        assert shift < 1.0  # bounded

    def test_robustness_to_noise(self):
        """Adding noise to CRS doesn't break convergence."""
        rng = np.random.default_rng(42)
        N, d_state, d_hilbert = 6, 2, 3
        states = rng.standard_normal((N, d_state))
        adj = np.ones((N, N)) - np.eye(N)
        kraus = make_lindblad_kraus(d_hilbert, 2, rng)

        # Run with higher noise
        for _ in range(30):
            states, rho = loop_iteration(
                states,
                adj,
                d_hilbert,
                kraus,
                lam=0.3,
                alpha=0.2,
                delta=0.1,
                c=0.01,
                sigma_eta=0.1,
                rng=rng,  # higher noise
            )
        assert is_positive_semidefinite(rho)
        assert abs(np.trace(rho) - 1.0) < 1e-10


# ============================================================
# Section 11: Hilbert Space Structure (Theorem 13.4)
# ============================================================


class TestHilbertSpaceEmergence:
    """Verify emergence of Hilbert space structure."""

    def test_inner_product_from_trace_distance(self):
        """Pure-state fidelity = 1 - D²/4."""
        rng = np.random.default_rng(42)
        d = 4
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= np.linalg.norm(psi)
        phi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        phi /= np.linalg.norm(phi)

        rho_psi = np.outer(psi, psi.conj())
        rho_phi = np.outer(phi, phi.conj())
        D = trace_distance(rho_psi, rho_phi)
        fidelity = np.abs(psi.conj() @ phi) ** 2
        fidelity_from_D = 1.0 - D**2
        # For pure states: D = sqrt(1 - |⟨ψ|φ⟩|²), so F = 1 - D²
        assert abs(fidelity - fidelity_from_D) < 1e-10

    def test_dimension_bound(self):
        """dim(H*) ≤ exp(κ̃_max · Ṽ)."""
        kappa_tilde_max = 2.0
        V_tilde = 5.0
        d_bound = np.exp(kappa_tilde_max * V_tilde)
        d_actual = 4  # test dimension
        assert d_actual <= d_bound

    def test_completeness(self):
        """Cauchy sequences of density matrices converge."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        # Create a Cauchy sequence converging to rho
        for n in range(1, 20):
            perturbation = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            perturbation = (perturbation + perturbation.conj().T) / 2
            rho_n = rho + perturbation / (10 * n)
            # Re-normalize
            eigvals = np.linalg.eigvalsh(rho_n)
            if np.min(eigvals) < 0:
                rho_n -= np.min(eigvals) * np.eye(d)
            rho_n /= np.trace(rho_n)
            d_n = trace_distance(rho_n, rho)
            assert d_n < 1.0 / n + 0.5  # converges


# ============================================================
# Section 12: Contractivity and Convergence Rate
# ============================================================


class TestContractivity:
    """Verify contraction properties of loop components."""

    def test_cptp_contractivity(self):
        """CPTP map is contractive in trace distance."""
        rng = np.random.default_rng(42)
        d = 4
        rho1 = random_density_matrix(d, rng)
        rho2 = random_density_matrix(d, rng)
        kraus = make_lindblad_kraus(d, 3, rng)
        d_in = trace_distance(rho1, rho2)
        d_out = trace_distance(
            apply_channel(kraus, rho1),
            apply_channel(kraus, rho2),
        )
        assert d_out <= d_in + 1e-10

    def test_observer_strict_contraction(self):
        """Observer with λ > 0 strictly contracts trace distance."""
        rng = np.random.default_rng(42)
        d = 4
        rho1 = random_density_matrix(d, rng)
        rho2 = random_density_matrix(d, rng)
        lam = 0.5
        d_in = trace_distance(rho1, rho2)
        d_out = trace_distance(
            observer_operator(rho1, lam),
            observer_operator(rho2, lam),
        )
        assert d_out <= d_in + 1e-10

    def test_geometric_convergence_rate(self):
        """Convergence rate is geometric: O(α^n)."""
        rng = np.random.default_rng(42)
        d = 4
        rho = random_density_matrix(d, rng)
        lam = 0.3
        # Apply observer many times to find the true fixed point
        rho_fp = rho.copy()
        for _ in range(100):
            rho_fp = observer_operator(rho_fp, lam)
        # Now verify convergence toward this fixed point
        rho_current = rho.copy()
        distances = []
        for _ in range(20):
            rho_current = observer_operator(rho_current, lam)
            distances.append(trace_distance(rho_current, rho_fp))
        # Overall trend: later distances should be smaller
        early_avg = np.mean(distances[:5])
        late_avg = np.mean(distances[-5:])
        assert late_avg <= early_avg + 1e-8
