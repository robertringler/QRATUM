# Value-asset analysis — reproducible scripts

Scripts backing `QRATUM_VALUE_ASSET_ANALYSIS.md`, `QRATUM_VALUE_MODEL.md`,
`QRATUM_COMPETITIVE_SUBSTITUTION.md`, `QRATUM_DESTRUCTION_TEST.md` and
`QRATUM_ASSET_INVENTORY.csv`. Run all of them from the repository root.

| Script | Backs | What it does |
|---|---|---|
| `qradle_capability_probe.py` | §4, §10 | Executes 9 adversarial probes against the live QRADLE engine and Merkle chain: replay stability of the audit root, detection of a non-deterministic executor, in-place tamper detection, whole-chain forgery, whether `get_proof`/`verify_proof` is a real inclusion proof, authorizer identity, per-level approval enforcement, rollback state restoration, and third-party verifiability. Requires nothing beyond the stdlib. |
| `ahtc_k_claim_probe.py` | §6.3 | Derives the arithmetic behind the registered `ahtc-k.scheduler.10x` claim from `performance::workload`'s key-space size (`COLD_POOL = 32`), showing the reduction ratio is `N/33` and the 10× gate cannot fail above N=330. Pure arithmetic, no dependencies. |
| `qratum_value_scorecard.py` | §17, §22 | Scores 14 candidate assets on 10 dimensions, computes weighted rankings under 5 weightings, prints the rank-sensitivity matrix, and models value concentration at three convexity levels. `--csv` emits the scorecard as CSV. |

```bash
python tools/value_analysis/qradle_capability_probe.py        # needs repo root on sys.path
python tools/value_analysis/ahtc_k_claim_probe.py
python tools/value_analysis/qratum_value_scorecard.py
python tools/value_analysis/qratum_value_scorecard.py --csv
```

To reproduce the Rust-side evidence directly:

```bash
cd os/qratum-os/crates/qratum-arbiter
cargo test --target x86_64-unknown-linux-gnu            # 163 passed, 0 failed
cargo build --tests                                     # fails: the crate's .cargo/config.toml
                                                        # forces target=x86_64-pc-windows-msvc
```

Scores in `qratum_value_scorecard.py` are analytical estimates anchored to the
`[OBSERVED]` and `[TESTED]` findings in the reports. The weightings are in the
source so they can be changed; the conclusion's robustness is the point of
having five of them.
