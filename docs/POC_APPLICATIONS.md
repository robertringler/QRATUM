# Physical Operator Computer — Application Domain Specifications and Benchmark Programs (POC-APPS-1.0)

**QRATUM Research Series — Applied Domain Analysis**

*Status:* Engineering specification, pre-M0 design
*Companion volumes:* POC-MONO-1.0 (architecture), POC-MONO-2.0 (theory), POC-MONO-3.0 (complexity)

---

## Purpose and Scope

Volume III established the POC's honest advantage envelope:

$$\text{dense operator} \;\wedge\; P \le 7\text{ bits} \;\wedge\; R \gtrsim R_{\text{amort}} \;\wedge\; \text{verification amortized within epoch}$$

This document translates that envelope into concrete engineering specifications for the three domains that satisfy all four conditions today, without waiting for fault-tolerant quantum hardware. Each domain section provides:

1. The operator formulation — exactly how the domain maps to POC primitives
2. QIL intent definitions for QRATUM execution
3. A concrete benchmark program with data model, success criteria, and baselines
4. A hardware phase map (which milestone enables what)
5. An honest feasibility assessment naming conditions of failure

The three domains in priority order: **(1) Linear Reservoir Computing**, **(2) Radar Direction-of-Arrival via MUSIC**, **(3) Massive MIMO Zero-Forcing Precoding**.

A fourth section treats **anti-pattern domains** — where the envelope fails and the POC should not be used — because knowing where not to build is as important as knowing where to build.

---

## Domain 1 — Linear Reservoir Computing

### 1.1 Why this is the structurally correct starting point

Volume III Theorem 29 established that the POC's recirculating loop registers are a **physically realized linear reservoir**: a fixed, dense, randomly initialized recurrent operator driven by an input signal. This is not an approximation of reservoir computing — it is the definition. Echo-state networks (Jaeger 2001) and liquid-state machines (Maass et al. 2002) require exactly this: a high-dimensional, fixed, contractive dynamical system applied to an input stream, with a trained linear readout. The POC's loop register applies the reservoir matrix W_res in one transit per time step.

There is no mapping overhead, no precision mismatch argument, no problem-to-architecture translation. The problem IS the machine.

### 1.2 Operator formulation

The discrete-time echo-state network equation:

$$\mathbf{x}(t+1) = (1-\alpha)\,\mathbf{x}(t) + \alpha\,f\bigl(W_{\text{res}}\,\mathbf{x}(t) + W_{\text{in}}\,\mathbf{u}(t) + \mathbf{b}\bigr)$$

where $\mathbf{x}(t) \in \mathbb{R}^n$ is the reservoir state, $\mathbf{u}(t) \in \mathbb{R}^m$ is the input, $W_{\text{res}} \in \mathbb{R}^{n \times n}$ is the fixed reservoir matrix, $\alpha$ is the leaking rate, and $f$ is a nonlinear activation. The **linear reservoir** (POC's native mode) sets $f = \text{id}$, yielding:

$$\mathbf{x}(t+1) = \tilde{W}\,\mathbf{x}(t) + W_{\text{in}}\,\mathbf{u}(t) \quad \text{where } \tilde{W} = (1-\alpha)\,I + \alpha\,W_{\text{res}}$$

The composite map $\tilde{W}$ is dense, fixed, applied once per time step, and must be contractive: $\rho(\tilde{W}) < 1$ (the echo-state property, equivalent to the spectral radius condition). The POC is configured as:

- **OAU**: configured to implement $\tilde{W}$ (SVD-compiled Clements mesh, held resident)
- **Modulator bank**: injects $W_{\text{in}}\,\mathbf{u}(t)$ as a correction to the loop field
- **Loop register**: holds $\mathbf{x}(t)$; one transit per time step applies $\tilde{W}$
- **Readout**: $\hat{y}(t) = W_{\text{out}}\,\mathbf{x}(t)$ where $W_{\text{out}} \in \mathbb{R}^{k \times n}$ is trained classically (ridge regression)

**Resource accounting**: one transit per time step = 2 ns per step. For a 1 kHz time series (audio, vibration, sensor): 10⁶ time steps/second = 2 ms of optical transit time, with the remaining ~999 ms idle — duty cycle too low to amortize the thermal floor unless the signal is high-frequency. For 100 kHz signals (radar returns, ultrasound, vibration monitoring): duty cycle becomes practical. The sweet spot is **high-frequency streaming signals where the 2 ns transit is the computational bottleneck**.

### 1.3 QIL intent

```qil
INTENT reservoir_time_series {
    OBJECTIVE classify_time_series
    HARDWARE ONLY POC NOT GPU NOT CPU
    CAPABILITY linear_reservoir
    CAPABILITY streaming_inference
    CONSTRAINT POC_MODES >= 64
    CONSTRAINT PRECISION_BITS <= 7
    CONSTRAINT REUSE_FACTOR >= 1000
    CONSTRAINT SPECTRAL_RADIUS < 1.0
    TIME deadline: 10us
    AUTHORITY user: signal_processing_lead
    TRUST level: verified
}
```

### 1.4 OperatorContract specification

```python
@dataclass(frozen=True)
class ReservoirOperatorContract(BaseContract):
    # The reservoir matrix W_res (n x n, dense, compiled to Clements angles)
    reservoir_angles: tuple[float, ...]    # Clements decomposition of W_tilde
    spectral_radius: float                 # rho(W_tilde) < 1 enforced
    leaking_rate: float                    # alpha in (0, 1]
    input_projection: tuple[float, ...]    # W_in, (n x m) matrix, flattened
    # POC resource bounds
    norm_bound: float                      # max gain, <= 1.0 for contractivity
    precision_bits: int                    # target: 6-7
    confidence: float                      # 0.99
    shot_budget: int                       # derived: 4^precision_bits per step
    fidelity_floor: float                  # 0.95
    calibration_epoch: str                 # hash of calibration state
    # Domain-specific
    time_step_ns: float                    # 2.0 (one transit)
    steps_per_epoch: int                   # reuse before recalibration check
```

### 1.5 Benchmark program: BM-R1 (NARMA-10 time-series identification)

**Task.** Identify the 10th-order Nonlinear Autoregressive Moving Average (NARMA-10) system:
$$y(t+1) = 0.3\,y(t) + 0.05\,y(t)\sum_{i=0}^{9}y(t-i) + 1.5\,u(t-9)\,u(t) + 0.1$$
where $u(t) \sim \text{Uniform}[0, 0.5]$. This is a standard reservoir computing benchmark with a known classical baseline.

**Data model.** Input: $10^4$ time steps, $u(t)$ sampled uniformly. Training split: first 5,000 steps (discard first 200 as washout). Test split: remaining 4,800 steps. This is entirely synthetic — reproducible with `numpy.random.seed(42)`.

**POC execution.** Reservoir size $n = 64$. $W_{\text{res}}$ initialized as random Gaussian $\mathcal{N}(0, 1/n)$, scaled so $\rho(W_{\text{res}}) = 0.9$. $\alpha = 0.3$. $W_{\text{in}} \in \mathbb{R}^{64 \times 1}$ random. $W_{\text{out}}$ trained by ridge regression on the collected reservoir states.

**Success criteria.**
- Primary: NMSE (normalized mean squared error) $\leq 0.03$ (classical software reservoir at $n=64$ achieves $\approx 0.01$–$0.02$; the POC is allowed $3\times$ degradation from precision loss)
- Secondary: energy per time step $\leq 50$ nJ (vs. $\approx 200$ nJ for GPU INT8 at comparable n)
- Tertiary: the execution log must pass statistical replay at 99% confidence

**Classical baseline.** NumPy dense matrix-vector product at fp32, same $n=64$, same $W_{\text{res}}$, running on host CPU. No GPU (n=64 is below GPU parallelism threshold; the comparison must be honest).

**Phase gate.** Executable at M0 ($n = 16$: run BM-R1-small at $n=16$ as a sanity check), primary target at M1 ($n=32$), full benchmark at M2 ($n=64$).

**Condition of failure.** If the calibration epoch drift causes NMSE to drift above 0.05 during a 10,000-step run without triggering a sentinel, the verification discipline has failed — this matters more than the NMSE number itself.

---

## Domain 2 — Radar Direction-of-Arrival via MUSIC

### 2.1 Problem setting

A uniform linear array (ULA) of $n$ antenna elements receives signals from $K < n$ narrowband sources at unknown angles $\theta_1, \ldots, \theta_K$. The MUSIC (MUltiple SIgnal Classification) algorithm estimates those angles from the spatial covariance matrix of the received array snapshot vector $\mathbf{x}(t) \in \mathbb{C}^n$:

$$R = \mathbb{E}[\mathbf{x}\mathbf{x}^\dagger] = \sum_{k=1}^K \sigma_k^2 \mathbf{a}(\theta_k)\mathbf{a}(\theta_k)^\dagger + \sigma_n^2 I$$

where $\mathbf{a}(\theta) = [1, e^{i\pi\sin\theta}, \ldots, e^{i(n-1)\pi\sin\theta}]^\top$ is the steering vector. The bottleneck is **eigendecomposing $R$** to separate the signal subspace (dominant $K$ eigenvectors) from the noise subspace (remaining $n-K$ eigenvectors), then sweeping the MUSIC pseudospectrum:

$$P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{a}(\theta)^\dagger \hat{U}_n \hat{U}_n^\dagger \mathbf{a}(\theta)}$$

where $\hat{U}_n$ is the estimated noise subspace.

### 2.2 Operator formulation

The covariance matrix $R$ is:
- **Dense**: full spatial correlation across all $n(n+1)/2$ pairs (for a ULA with uncorrelated sources)
- **Hermitian positive-definite**: directly configurable as an OAU operator (self-adjoint $\Rightarrow$ real eigenvalues, physical spectrum)
- **Fixed within a coherent processing interval (CPI)**: the same $R$ is estimated from hundreds of pulses over $\sim$10–100 ms and then decomposed — this is the reuse window

**POC execution flow:**

1. **Classical pre-processing** (host): collect $M$ snapshot vectors $\mathbf{x}(t)$, form sample covariance $\hat{R} = \frac{1}{M}\sum_{t=1}^M \mathbf{x}(t)\mathbf{x}(t)^\dagger$. Cost: $O(Mn^2)$ MACs, done on GPU/DSP in parallel with last-CPI processing.
2. **SVD-compile** (host, $O(n^3)$): factor $\hat{R} = U\Lambda U^\dagger$ (classical); compile $U$ to Clements angles for the OAU.
3. **Configure OAU** (10 µs thermal): load the Clements angles for $U$.
4. **Ringdown spectroscopy** (POC Spectral Unit): inject a broadband probe state; the ringdown beat spectrum yields all $n$ eigenvalues $\lambda_1 \geq \ldots \geq \lambda_n$ in $O(1/\Delta\lambda_{\min})$ transits.
5. **Noise-subspace projection** (POC or host): compute $\hat{U}_n$ from the bottom $n-K$ eigenvectors; evaluate $P_{\text{MUSIC}}(\theta)$ over a steering-vector grid (host, $O(n \cdot |\Theta|)$).
6. **Peak detection** (host): locate angle estimates.

**The POC's role is step 4** — replacing the $O(n^3)$ classical EVD with the spectral readout primitive. **Steps 1–3 and 5–6 remain classical.** The POC accelerates only the eigendecomposition; the benefit is energy (Θ(n) vs. Θ(n³) classical EVD at moderate precision) and latency on the critical path.

**Precision requirement.** MUSIC requires resolving the $K$ signal eigenvalues from the noise floor $\sigma_n^2$. For SNR = 10 dB: eigenvalue ratio $\lambda_K / \lambda_{K+1} \approx 10$, requiring $\log_2(10) \approx 3.3$ bits to separate. For SNR = 20 dB: $\approx 6.6$ bits. The POC's 5–7 bit certified precision covers SNR up to $\sim$20 dB. **At higher SNR, the precision constraint binds and the POC requires digital cleanup.**

### 2.3 QIL intent

```qil
INTENT radar_music_doa {
    OBJECTIVE estimate_direction_of_arrival
    HARDWARE ONLY POC NOT GPU
    CAPABILITY spectral_decomposition
    CAPABILITY ringdown_eigenspectrum
    CONSTRAINT POC_MODES >= 64
    CONSTRAINT PRECISION_BITS <= 7
    CONSTRAINT SNR_DB <= 20
    CONSTRAINT SOURCES < 32
    TIME deadline: 1000us
    AUTHORITY user: radar_operator
    TRUST level: verified
}
```

### 2.4 OperatorContract specification

```python
@dataclass(frozen=True)
class MUSICOperatorContract(BaseContract):
    # The covariance operator (configured as OAU self-adjoint map)
    covariance_clements_angles: tuple[float, ...]  # U factor of R = U Lambda U†
    covariance_eigenvalue_range: tuple[float, float]  # (lambda_min, lambda_max)
    num_sources: int                               # K, needed for subspace split
    # Spectral Unit configuration
    loop_gain: float                               # alpha < 1/rho(R), resolvent
    ringdown_duration_ns: float                    # T_obs; sets Delta lambda_min
    # POC resource bounds
    norm_bound: float                              # ||R|| <= this
    precision_bits: int                            # 6-7 for SNR <= 20 dB
    confidence: float
    shot_budget: int
    fidelity_floor: float                          # 0.95 on eigenvalue estimates
    calibration_epoch: str
    spectral_gap_min: float                        # lambda_K - lambda_{K+1} >= this
```

### 2.5 Benchmark program: BM-D1 (ULA direction-of-arrival, n=64)

**Task.** Estimate the angles of $K=2$ sources impinging on a 64-element ULA with half-wavelength spacing. Sources at $\theta_1 = 15°, \theta_2 = 25°$ (angular separation 10°, chosen to be above the classical Rayleigh limit at $n=64$: $\Delta\theta_{\min} \approx 0.9°$).

**Data model.** Monte Carlo over 500 independent CPI realizations. Per CPI: $M = 200$ snapshots. Source powers: $\sigma_1^2 = \sigma_2^2 = 1$. Noise power $\sigma_n^2$ swept over SNR $\in \{5, 10, 15, 20\}$ dB. Data generated from the standard complex Gaussian model.

**POC execution.** For each CPI: form $\hat{R}$ classically; compile to Clements angles; execute ringdown spectroscopy for eigenspectrum; extract $\hat{U}_n$; sweep pseudospectrum on host.

**Success criteria.**
- Primary: RMSE of angle estimates $\leq 1.5°$ at SNR = 15 dB, $n=64$, $K=2$ (classical MUSIC at fp64 achieves $\approx 0.3°$; the POC is allowed $5\times$ degradation from precision loss; if it exceeds $1.5°$ the precision is insufficient)
- Secondary: eigenvalue estimation error $|\hat{\lambda}_i - \lambda_i| / \lambda_i \leq 5\%$ for all $i$ at SNR = 15 dB
- Tertiary: energy per CPI eigendecomposition $\leq 100$ nJ (vs. $\approx 4$ µJ for Intel MKL DSYEV at $n=64$ on a server-class CPU)
- Adversarial sub-benchmark: at SNR = 5 dB, RMSE is expected to degrade past the $1.5°$ criterion — **this must be reported, not excluded**. The POC fails gracefully at low SNR; the classical baseline does not fail gracefully (it fails differently). Report both failure modes.

**Classical baseline.** LAPACK DSYEV (symmetric EVD) on a single CPU core, $n=64$, same $\hat{R}$. Runtime and energy logged via PAPI hardware counters.

**Phase gate.** BM-D1-small at $n=16$ executable at M0 ($K=1$ source, SNR=20 dB only); BM-D1 at $n=64$ executable at M2.

**Condition of failure.** If the ringdown spectral resolution $\Delta\lambda_{\min} > \lambda_K - \lambda_{K+1}$ (eigenvalue gap smaller than resolution), the ringdown cannot distinguish the signal from noise subspace — the primary benchmark fails for physical reasons, not calibration. This is the spectral-resolution limit (Volume II Theorem 6; Vol III Theorem 19). The experiment must measure $\Delta\lambda_{\min}$ directly and report whether the gap condition is satisfied.

---

## Domain 3 — Massive MIMO Zero-Forcing Precoding

### 3.1 Problem setting

A base station with $n_{\text{BS}}$ antennas simultaneously serves $n_{\text{UE}}$ single-antenna users in the downlink. The $n_{\text{UE}} \times n_{\text{BS}}$ channel matrix $H$ is estimated per coherence interval (~1–10 ms). The zero-forcing precoder is:

$$W_{\text{ZF}} = H^\dagger \bigl(H H^\dagger\bigr)^{-1}$$

Applied to the transmit symbol vector $\mathbf{s} \in \mathbb{C}^{n_{\text{UE}}}$, the precoded signal is $\mathbf{x} = W_{\text{ZF}} \mathbf{s}$, achieving perfect inter-user interference nulling in the noise-free limit.

The bottleneck is forming and inverting $G = H H^\dagger$ ($n_{\text{UE}} \times n_{\text{UE}}$ Gram matrix), which must be done at every coherence interval. For massive MIMO ($n_{\text{BS}} = 64$, $n_{\text{UE}} = 8$): $G \in \mathbb{C}^{8 \times 8}$, trivially inverted classically. The POC is NOT competitive at $n_{\text{UE}} = 8$. The relevant regime is **$n_{\text{UE}} = n_{\text{BS}} / 2 = 32$**: a fully-loaded massive MIMO system where $G$ is $32 \times 32$ and the inversion and application cost is non-trivial in real-time.

### 3.2 Operator formulation

The ZF precoding step is $W_{\text{ZF}} \mathbf{s} = H^\dagger G^{-1} \mathbf{s}$ where $G = HH^\dagger + \lambda I$ ($\lambda$ is a regularization/noise floor term). This decomposes into:

1. **Step A**: solve $G \mathbf{v} = \mathbf{s}$, i.e., compute $\mathbf{v} = G^{-1} \mathbf{s}$
2. **Step B**: compute $\mathbf{x} = H^\dagger \mathbf{v}$ (classical, $O(n_{\text{BS}} n_{\text{UE}})$ MACs)

**Step A is the POC's role** — via the resolvent mode (Neumann series) of the Spectral Unit:
$$(I - \alpha G) \mathbf{x} = \alpha \mathbf{s} \quad \Rightarrow \quad \mathbf{x} = (I - \alpha G)^{-1} \alpha \mathbf{s} = \sum_{k=0}^\infty \alpha^k G^k (\alpha \mathbf{s})$$

For $\alpha < 1/\|G\|$, this converges geometrically. Each term is one loop transit. Convergence rate: $(\alpha \|G\|)^k$ per transit.

**Why this fits the envelope:**
- **Dense**: $G = HH^\dagger$ is dense (full-rank covariance for a rich scattering channel)
- **Low precision**: the ZF precoder's spectral efficiency degradation from 6-bit vs. fp32 computation is $< 0.5$ dB in realistic channels (verified in simulation by Björnson et al. 2017)
- **High reuse**: within one coherence interval ($\sim$1 ms), the same $G$ is applied to $n_{\text{UE}}$ transmit symbol vectors per OFDM subcarrier, across hundreds of subcarriers → thousands of resolvent applications before $G$ changes
- **Reconfiguration**: $G$ changes at coherence time (~1 ms); reconfiguration cost (10 µs) is 1% of the coherence window — acceptable

### 3.3 QIL intent

```qil
INTENT mimo_zf_precoder {
    OBJECTIVE compute_zf_precoder_output
    HARDWARE ONLY POC NOT CPU
    CAPABILITY resolvent_mode
    CAPABILITY neumann_series
    CONSTRAINT POC_MODES >= 32
    CONSTRAINT PRECISION_BITS <= 7
    CONSTRAINT SPECTRAL_EFFICIENCY_LOSS_DB <= 0.5
    CONSTRAINT COHERENCE_INTERVAL_MS >= 1.0
    TIME deadline: 100us
    AUTHORITY user: baseband_controller
    TRUST level: verified
}
```

### 3.4 OperatorContract specification

```python
@dataclass(frozen=True)
class MIMOPrecoderContract(BaseContract):
    # Gram matrix operator G = H H† + lambda I (n_UE x n_UE, configured on OAU)
    gram_clements_angles: tuple[float, ...]    # SVD-compiled Clements program for G
    regularization: float                     # lambda (noise floor)
    # Resolvent configuration
    loop_gain: float                          # alpha < 1/||G||; set by norm_bound
    convergence_tolerance: float              # ||residual|| / ||rhs|| < this
    max_transits: int                         # loop depth limit = loop lifetime
    # POC resource bounds
    norm_bound: float                         # ||G|| upper bound (from channel estimate)
    precision_bits: int
    confidence: float
    shot_budget: int
    fidelity_floor: float
    calibration_epoch: str
    # MIMO-specific
    num_users: int                            # n_UE
    num_antennas: int                         # n_BS
    subcarriers_per_epoch: int               # reuse count within one coherence interval
```

### 3.5 Benchmark program: BM-M1 (massive MIMO ZF precoder, n_BS=64, n_UE=32)

**Task.** Compute the ZF precoded downlink signal for $n_{\text{UE}} = 32$ users from a $n_{\text{BS}} = 64$ antenna base station, over 100 OFDM subcarriers, for one coherence interval.

**Channel model.** 3GPP Urban Macro (UMa), 3.5 GHz, 100 m inter-site distance, 32 users uniformly distributed in a 120° sector. Correlated Rayleigh fading with spatial correlation given by the Kronecker model. Channel matrix $H$ has full rank with high probability.

**Data model.** 500 independent channel realizations. Per realization: 100 subcarriers, each requiring one $G$ inversion (the same $G$ per subcarrier within a coherence interval is the high-reuse assumption; in frequency-selective channels $G$ varies per subcarrier — test both: flat-fading (high reuse) and mildly frequency-selective (moderate reuse)).

**POC execution.** Form $G = HH^\dagger + \lambda I$ classically; compute $\|G\|$ and set $\alpha = 0.9/\|G\|$; compile to Clements angles; configure OAU; run resolvent loop until convergence tolerance met; measure $\mathbf{v}$; compute $\mathbf{x} = H^\dagger \mathbf{v}$ classically.

**Success criteria.**
- Primary: spectral efficiency within 0.5 dB of classical ZF (fp64) at SNR = 15 dB, averaged over 500 channel realizations
- Secondary: energy per subcarrier precoding step $\leq 50$ nJ (vs. $\approx 2$ µJ for CPU LAPACK ZGESV at $n=32$)
- Tertiary: convergence within 40 transits (the loop depth limit from Volume I's insertion-loss model; if convergence requires more, the algorithm must fall back to digital cleanup iteration)
- Adversarial: for ill-conditioned $G$ ($\kappa(G) > 100$, occurring when users cluster spatially), report convergence failure explicitly — this is BM-6's MIMO analog (designed-to-lose at high condition number)

**Classical baseline.** LAPACK ZGESV ($n=32$ complex linear solve) on a single CPU core, Intel MKL.

**Phase gate.** BM-M1-small ($n_{\text{BS}}=16$, $n_{\text{UE}}=8$) at M0; full BM-M1 at M2.

**Condition of failure.** If the resolvent loop fails to converge within 40 transits for more than 10% of channel realizations (i.e., $\kappa(G) > (0.9)^{-40} \approx 77$ too often for the given channel model), the precoder falls back to the hybrid digital iteration loop. The benchmark must report the fallback rate — this is honest performance characterization, not a disqualifying failure.

---

## Cross-domain benchmark schedule

| Benchmark | Domain | Phase | n | Primary metric | Expected outcome |
|---|---|---|---|---|---|
| BM-R1-small | Reservoir | M0 (n=16) | 16 | NMSE ≤ 0.1 | Sanity check; verification discipline primary |
| BM-D1-small | Radar DOA | M0 (n=16) | 16 | RMSE ≤ 3° @ 20 dB | Spectral readout validation |
| BM-R1 | Reservoir | M1 (n=32) | 32 | NMSE ≤ 0.05 | Energy advantage expected |
| BM-M1-small | MIMO | M1 (n=32, UE=8) | 32 | SpEff within 1 dB | Resolvent convergence validation |
| BM-R1-full | Reservoir | M2 (n=64) | 64 | NMSE ≤ 0.03 | Main reservoir benchmark |
| BM-D1 | Radar DOA | M2 (n=64) | 64 | RMSE ≤ 1.5° @ 15 dB | Main radar benchmark |
| BM-M1 | MIMO | M2 (n=64, UE=32) | 64 | SpEff within 0.5 dB | Main MIMO benchmark |

Each benchmark result is published with:
- Contract hash
- Calibration epoch hash
- Shot budgets per run
- Event log digest (sufficient for statistical replay)
- The adversarial sub-result (designed-to-lose scenario)

A benchmark number without a replayable log entry **does not exist** (Volume I rule, enforced here).

---

## Anti-pattern domains (and why)

These are domains that superficially look like they fit but fail one or more envelope conditions.

### AP-1: Large language model inference

**Why it looks right.** Dense weight matrices, INT4/INT8 standard in post-training quantization, extreme reuse (same weights, billions of tokens). Every condition looks satisfied.

**Why it fails.** The `n` constraint. A single attention head in GPT-4 class models uses $d_{\text{head}} = 128$ (or 64 in some architectures). The M2 spec is $n = 64$. A single transformer layer has hundreds of heads plus FFN matrices of dimension 4096–16384. Tiling into $64 \times 64$ blocks requires $\sim$$10^3$–$10^4$ POC calls per layer with classical communication between them. The inter-call data movement cost eliminates the energy advantage. **At M4 ($n = 256+$), single-head matrices become feasible**; at $n = 4096$ (10+ years out), the full layer fits. This domain is a valid long-range target but not a near-term one.

**Kill criterion.** POC applicable to LLM inference iff $n \geq d_{\text{model}}/\text{attention\_heads}$ without tiling. At current model sizes, $n \geq 512$ minimum.

### AP-2: Graph neural networks / PageRank on web-scale graphs

**Why it looks right.** Operator applied repeatedly to a vector (power iteration). Structural similarity to the reservoir or DOA problem.

**Why it fails C1.** Real-world graphs have adjacency matrices with density $\sim$$10^{-9}$ (Facebook: $10^9$ edges / $(10^9)^2$ nodes). Classical sparse matvec is $O(\text{nnz}) = O(n)$. The POC costs $\Theta(n)$ energy per transit regardless of sparsity — it applies the full $n \times n$ operator even if $n-1$ of $n^2$ entries are zero. For $n = 10^9$ and density $10^{-9}$, classical is $\sim$$10^9$ operations vs. the POC's $\sim$$10^{18}$. **Sparse operators are the worst-case for the POC.** This is Attack 4 from Volume III (quantified: advantage vanishes when nnz $= O(n)$).

**Kill criterion.** POC applicable to graph problems iff the graph's adjacency matrix is dense (complete or near-complete subgraphs). Social/web graphs: never. Dense knowledge graphs or small coupling matrices: potentially.

### AP-3: Scientific computing at high precision

**Why it looks right.** Many scientific codes have dense operators (climate spectral models, finite-element method assembled matrices, DFT $k$-point matrices) applied repeatedly.

**Why it fails C2.** Scientific computing typically requires $10^{-8}$–$10^{-14}$ relative error (fp32 to fp64). The POC's fundamental precision ceiling at $\sim$23 bits (the shot-noise SQL at µJ/mode) would suffice for fp32-equivalent, but reaching it requires $4^{23} \approx 7 \times 10^{13}$ shots per amplitude estimate. The hybrid iteration loop (Volume I §8.4) can in principle reach fp32-equivalent via digital cleanup, but each cleanup round costs a full digital-precision residual computation — at that point the host's fp64 compute dominates and the POC offers no advantage.

**Kill criterion.** POC applicable iff the required relative error $\geq 10^{-2}$ (i.e., about 7 bits). High-precision scientific computing: not applicable.

### AP-4: Quantum chemistry

**Why it looks right.** Hamiltonian matrix-vector products in iterative solvers (Lanczos, Davidson); dense in the molecular orbital basis; repeated application.

**Why it fails C2 catastrophically.** Chemical accuracy is 1 kcal/mol ≈ $1.6 \times 10^{-3}$ Hartree. A typical molecular energy is $\sim$$10^2$ Hartree, so required relative precision is $\sim$$10^{-5}$ = about 17 bits. This is the third regime where C2 fails hard.

---

## The M0 experiment: what to actually build first

Given the above, the M0 ($n = 8$–$16$) hardware bring-up should focus on **BM-R1-small** as the primary validation experiment, with **BM-D1-small** as the spectral-readout verification. The reasons:

1. **BM-R1-small is self-contained**: the reservoir matrix is entirely under the experimenter's control (no external data needed), and the NARMA-10 task has well-characterized classical baselines at every precision level. The experiment can be run with synthetic data and the result compared to a mathematical expected value.

2. **BM-D1-small validates the spectral readout primitive** specifically — it exercises the ringdown mode and the first-class spectral output that is the POC's most distinctive hardware feature. Without this working, none of the other benchmarks are meaningful.

3. **The verification discipline is the primary M0 output.** Whether the NMSE is 0.08 or 0.12 matters less than whether: (a) the calibration epoch hash correctly predicts replay success/failure when controlled drift is injected; (b) the fidelity floor rejection works (the contract returns REJECT, not a silent wrong answer, when fidelity drops below threshold); (c) the event log is complete and contains everything needed for statistical replay by a party who did not run the experiment.

**If only one thing survives M0, it should be a working calibration-epoch gate.** That is the engineering contribution (Outcome B from Volume III) that no prior photonic computing system has provided.

---

## Research questions opened by this domain analysis

The application specifications above expose specific research questions that need answers before M1 can be designed with confidence:

**RQ-1 (Reservoir):** What is the accuracy loss from 5–7 bit precision in linear reservoirs as a function of reservoir dimension $n$ and spectral radius $\rho$? Published literature uses fp64; the POC precision model predicts graceful degradation for $\rho < 0.9$ but this has not been validated experimentally. Answerable in simulation before M0.

**RQ-2 (Radar):** Does the ringdown spectroscopy achieve eigenvalue resolution $\Delta\lambda_{\min} < \lambda_K - \lambda_{K+1}$ for typical radar covariance matrices at $n=64$? The theoretical limit is $\sim$7 bits of spectral resolution (Volume II, first-principles derivation). Typical eigenvalue gaps in ULA covariance matrices at SNR = 15 dB are $\sim$20 dB (factor of 10) — the POC needs $\log_2(10) \approx 3.3$ bits to resolve this, well within the 7-bit ceiling. But for scenarios with near-coherent sources (angular separation $\ll 1/n$), the gap closes and the resolution limit bites. Map the failure region before M1.

**RQ-3 (MIMO):** What fraction of realistic 3GPP channels produce $\kappa(G) > 77$ (the 40-transit resolvent convergence limit)? This sets the fallback rate for BM-M1 and determines whether the hybrid digital cleanup loop is needed for the majority or minority of realizations. Answerable in simulation from 3GPP channel model parameters.

**RQ-4 (Calibration amortization):** At what reuse factor $R$ does the Ω($n^2$) verification tomography cost fall below 1% of the total compute time? For BM-D1 (500 CPIs of 200 snapshots each): total executions = $10^5$; tomography is performed once per calibration epoch. If one epoch lasts $10^4$ executions, verification overhead is 1%. This is the regime where the audit is essentially free. Compute the break-even for each benchmark.

**RQ-5 (Cross-domain):** Are there compound workloads — e.g., a radar front-end feeding an echo-state classifier — where the POC can hold the operator configured for one task and share the physical state with a downstream reservoir step, eliminating a measure-and-reload round trip? This is the "operator fusion" question for heterogeneous workloads and is the natural M3 research question.

---

## Summary: the deployment decision tree

```
Is the operator dense?  (nnz = Omega(n^2))
  NO  → use classical sparse methods; POC not competitive
  YES ↓

Is required precision ≤ ~7 bits?
  NO  → high precision required; use classical fp64 or GPU
    (exception: use POC for the inner loop with digital cleanup
     only if the residual step is < 10% of total cost)
  YES ↓

Is the same operator applied to ≥ R_amort ≈ 10^2–10^3 distinct inputs?
  NO  → amortization condition fails; reconfig + verification overhead
         exceeds savings; use GPU or ASIC
  YES ↓

Is wall-clock latency or energy at the point of operation a binding constraint?
  NO  → batch on GPU; POC offers no distinct advantage
  YES ↓

DEPLOY POC
  Reservoir computing (streaming time-series): OAU + loop register as reservoir
  Radar/sonar (DOA, beamforming): OAU + ringdown spectral readout
  Massive MIMO (precoding solve): OAU + resolvent Neumann series
  Control (LQR/Riccati online solve): OAU + resolvent, high duty cycle
  Signal processing (covariance EVD, spectral filtering): OAU + SU
```

The tree has five decision nodes. Failing any one of them means the POC should not be used. Passing all five means the POC is the right accelerator for the problem — not because it computes something new, but because it applies a specific operator physically rather than numerically, which is faster and cheaper exactly in this regime.

---

*Document POC-APPS-1.0 · QRATUM Research Series*
*Companion to POC-MONO-1.0, POC-MONO-2.0, POC-MONO-3.0*
