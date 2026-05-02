# QRATUM Completeness Scorecard
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Score = `1 - (major+block findings) / (files in subtree)`. Heuristic: signals true *defects* per file, not feature completeness. A 100% score means no `major`/`block` static findings, NOT that every promised feature is built.

| Subsystem | Files | Major+Block | Score | Evidence |
|---|---|---|---|---|
| `qratum` | 113 | 2 | 98.2% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum` |
| `qratum-rust` | 29 | 1 | 96.6% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum-rust` |
| `qratum_platform` | 4 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_platform` |
| `qratum_platform_legacy` | 20 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_platform_legacy` |
| `qratum_ai_platform` | 93 | 1 | 98.9% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_ai_platform` |
| `qratum_aas` | 21 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_aas` |
| `qratum_asi` | 135 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_asi` |
| `qratum_chess` | 78 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_chess` |
| `qratum_desktop` | 20 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qratum_desktop` |
| `quasim` | 307 | 14 | 95.4% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `quasim` |
| `quasim_master_all.py` | 1 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `quasim_master_all.py` |
| `quasim_repo_enhancement.py` | 1 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `quasim_repo_enhancement.py` |
| `quasim_spacex_demo.py` | 1 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `quasim_spacex_demo.py` |
| `qagents` | 63 | 1 | 98.4% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qagents` |
| `qcore` | 7 | 2 | 71.4% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qcore` |
| `qnode` | 9 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qnode` |
| `qnx` | 15 | 3 | 80.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qnx` |
| `qnx_agi` | 26 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qnx_agi` |
| `qos` | 3 | 2 | 33.3% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qos` |
| `qmp` | 3 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qmp` |
| `qil` | 5 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qil` |
| `qdl` | 13 | 1 | 92.3% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qdl` |
| `qsk` | 33 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qsk` |
| `qtime` | 6 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qtime` |
| `qreal` | 19 | 2 | 89.5% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qreal` |
| `qscenario` | 20 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qscenario` |
| `qcampaign` | 6 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qcampaign` |
| `qconstitution` | 6 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qconstitution` |
| `qintervention` | 6 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qintervention` |
| `qledger` | 6 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qledger` |
| `qrVITRA` | 8 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `qrVITRA` |
| `Aethernet` | 31 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `Aethernet` |
| `aion` | 28 | 2 | 92.9% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `aion` |
| `omnilex` | 17 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `omnilex` |
| `epistemic_heat_sink` | 4 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `epistemic_heat_sink` |
| `topological_observer` | 3 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `topological_observer` |
| `spine` | 2 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `spine` |
| `xenon` | 82 | 1 | 98.8% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `xenon` |
| `manuscripts` | 26 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `manuscripts` |
| `dashboard` | 17 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `dashboard` |
| `dashboards` | 0 | 0 | N/A (path absent or empty) | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `dashboards` |
| `tests` | 287 | 4 | 98.6% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `tests` |
| `benchmarks` | 15 | 0 | 100.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `benchmarks` |
| `.github` | 7 | 3 | 57.1% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `.github` |
| `scripts` | 50 | 3 | 94.0% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `scripts` |
| `docs` | 265 | 7 | 97.4% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `docs` |
| `docs-site` | 36 | 4 | 88.9% | audit/code/findings.jsonl + audit/security/findings.jsonl filtered by path prefix `docs-site` |

## Caveats
- Subsystems claimed in problem statement but **absent** from filesystem are flagged 'N/A':
  - `dashboards`
