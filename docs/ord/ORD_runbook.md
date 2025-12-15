# Operational Readiness Demo (ORD) Runbook

## Overview

The Phase VI ORD validates QuASIM×QuNimbus hybrid stacks under a 72-hour burn-in. It streams Φ_QEVF telemetry alongside energy, fidelity, throughput, and entanglement yield across the us-east, us-west, eu-central, and ap-sg regions. This runbook captures deterministic procedures for starting, monitoring, and stopping the demo while maintaining compliance posture and reproducible artifacts.

## Prerequisites

- Kubernetes clusters available in target regions with GPU/DPU workloads enabled.
- Access to the quasim-operator CRDs (QuasimCluster, EntanglementLink, ValuationFeed).
- Service accounts with mTLS certificates and JWT signing keys provisioned.
- Prometheus and Grafana instances reachable from demo clusters.
- Access to the compliance CI pipelines and artifact storage (SBOM, MC/DC reports, ord canaries).
- Seeds recorded in `artifacts/metadata.json` for deterministic telemetry playback.

## Start Procedure

1. Validate infrastructure health via `scripts/ord/verify.sh --preflight`.
2. Deploy the latest container images referenced in `examples/overlays/ord/values.yaml` using Helm.
3. Apply QuasimCluster, EntanglementLink, and ValuationFeed CRDs for each region.
4. Launch the telemetry agent using `scripts/ord/start.sh --duration 72h`.
5. Confirm that Prometheus scrapes the telemetry endpoint and that Grafana dashboard `qevf_telemetry` renders Φ_QEVF, RMSE, and EP95 panels.
6. Trigger ORD GitHub Action workflow (`ord.yml`) to start six-hour canary cycles.
7. Begin continuous monitoring via the ORD Grafana playlist and alert routes.

## Stop Procedure

1. Gracefully stop telemetry via `scripts/ord/start.sh --stop` which drains publishers and flushes metrics.
2. Scale down QuasimCluster workloads per region and ensure entanglement links are quiesced.
3. Archive logs, SBOMs, compliance reports, and Grafana snapshots to `artifacts/ord/` with SHA256 manifests.
4. Revoke temporary credentials, rotate JWT signing keys, and confirm compliance attestations are uploaded.
5. Update ORD status badge in `README.md` to reflect completion state.

## Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation |
| --- | --- | --- |
| Φ_QEVF drops below 1.0e17 ops/kWh median | `scripts/ord/verify.sh` SLO check and Grafana alert | Trigger adaptive load shedding, rebalance entanglement links, rerun canary |
| Fidelity P95 < 0.997 | Prometheus alert `ord_fidelity_p95_breach` | Increase error correction depth, re-initialize affected qubits |
| Energy variance σ_E > 5% | Dashboard RMSE panel | Enable MERA compression compensators, recalibrate energy regulators |
| Telemetry agent crash | Kubernetes liveness probe failure | Restart pod with deterministic seed, inspect `/var/log/ord-agent.log` |
| Compliance pipeline failure | GitHub Action `compliance.yml` status | Re-run static analysis, update SBOM, remediate vulnerabilities |

## Service Level Objectives

- **Φ_QEVF median ≥ 1.0e17 ops/kWh** over rolling 4h window.
- **Fidelity P95 ≥ 0.997** per region.
- **Energy variance σ_E ≤ 5%** across racks.
- **Throughput ≥ target load** defined per rack (see `configs/ord/throughput_targets.yaml`).
- **Zero unresolved CRITICAL alerts** for > 15 minutes.

## Observability & Reporting

- Prometheus metrics exposed at `:9090/metrics` include `qevf_ops_per_kwh`, `energy_kwh`, `fidelity`, `throughput`, and `entanglement_yield`.
- Grafana dashboard JSON stored at `dashboards/grafana/qevf_telemetry.json` with panels for Φ_QEVF, RMSE, EP95, and region comparisons.
- ORD pipeline exports artifacts to `artifacts/ord/` with reproducibility metadata.

## Contacts & Escalation

- ORD Captain: <ord-captain@quasim.example.com>
- Compliance Lead: <compliance@quasim.example.com>
- Operator Hotline: +1-800-QUASIM-ORD
