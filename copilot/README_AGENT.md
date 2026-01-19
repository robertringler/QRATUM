# QuASIM Copilot Agent — Pilot Track (SpaceX/NASA)

## Overview

This document defines the operational purpose, workflow, and safeguards for the **QuASIM Copilot Agent** running under the **pilot/spacex-nasa** branch.  
The agent maintains deterministic, audit-ready code and outputs for the QuASIM × SpaceX/NASA demonstration.

---

## 1. Mission

The Copilot agent autonomously manages:

- Deterministic **demo generation** using `quasim_spacex_demo.py`
- Continuous Integration (**CI**) runs that validate reproducibility
- Artifact publishing (`spacex_demo_report.json`, `starship_demo_report.json`)
- Compliance and documentation enforcement (MC/DC, DO-178C alignment metadata)
- Creation of tagged demo releases (`pilot-v1.0`)

This branch represents QuASIM's **sanitized pilot track**—no proprietary kernels, no internal datasets, no secrets.

---

## 2. Key Responsibilities

| Function | Description |
|-----------|--------------|
| **Branch Governance** | Operates exclusively on `pilot/spacex-nasa`; forbidden to modify core runtime directories (`quasim/core`, `quasim/kernels`, etc.). |
| **Determinism Enforcement** | Ensures identical outputs across repeated runs by enforcing fixed seeds (`numpy` RNG 42). |
| **Runtime Validation** | Confirms total runtime under 60 s on `ubuntu-latest` CI runners. |
| **Artifact Management** | Uploads JSON reports as CI artifacts and attaches them to tagged GitHub releases. |
| **Documentation Updates** | Validates presence of README pilot section, Make targets, and legal placeholders. |
| **Security Posture** | No external network access or API calls; secret scanning enabled. |

---

## 3. Directory Map

```
copilot/
├─ agent_config.yaml       # Main configuration (guardrails, validation, runtime)
└─ README_AGENT.md         # This documentation

configs/meco_profiles/      # Demo shaping profiles (Falcon 9, Starship)
.github/workflows/          # CI workflow: spacex-demo.yml
legal/                      # NDA & LOI placeholders
```

---

## 4. Execution Flow

1. **Trigger:** push or workflow_dispatch → `spacex-demo.yml`
2. **Setup:** Python 3.11 + `numpy`, `matplotlib`
3. **Run:**  

   ```bash
   python quasim_spacex_demo.py --profile configs/meco_profiles/spacex_f9_stage1.json
   python quasim_spacex_demo.py --profile configs/meco_profiles/starship_hotstaging.json
   ```

4. **Validate:** identical JSON metrics across runs, runtime < 60 s
5. **Artifacts:** upload two JSON files + CI logs
6. **Release:** tag pilot-v1.0 with attached artifacts and badges

---

## 5. Compliance and Determinism Checks

| Check | Target |
|-------|--------|
| Reproducibility | Deterministic outputs verified by CI |
| MC/DC coverage reference | Placeholder metadata (demo only) |
| Security | No external calls, secrets scanned |
| Documentation | README pilot section present |
| Legal | NDA and LOI files exist in /legal/ |

---

## 6. Approved Dependencies

Only:

- `numpy`
- `matplotlib`

No quantum kernels, no JAX/PyTorch, no external APIs.

---

## 7. Release Policy

After CI success, the agent automatically creates tag `pilot-v1.0`.

Artifacts are attached to the release along with summary notes.

Each subsequent demo update increments the tag (v1.1, v1.2, …).

Proprietary content must never enter this branch.

---

## 8. Human Review Expectations

Maintainers should confirm:

1. CI completes in under one minute.
2. Reports contain deterministic `optimized_alpha` values.
3. No external network or kernel imports appear.
4. Legal placeholders remain intact.
5. Badges render correctly in README.

---

## 9. Contact

**QuASIM DevOps Lead:**  
<devops@quasim.io>

**Procurement / External Liaisons:**  
<procurement@quasim.io>

---

This document is internal to the QuASIM pilot branch and may be shared with partner reviewers (SpaceX/NASA) for transparency.
