# P0.4 — Reproducibility Contract

## Toolchain

- `rust-toolchain.toml` pins `nightly-2025-04-01`.
- `rust-src` component required (build-std).

## Required output set

After `tools/repro/reproduce.{ps1,sh}` completes, `out/repro/` MUST contain:

| File                         | Producer                          |
|------------------------------|-----------------------------------|
| `final_state.hash`           | `sha256sum target/.../qratum.efi` |
| `audit.hash`                 | `sha256sum out/repro/test_log.txt`|
| `replay_receipt.json`        | toolchain + bytes + hashes + UTC  |
| `performance_report.json`    | extracted "test result" lines     |

## Hardware envelope

- **CPU**: x86_64, ≥ 2 logical cores, invariant TSC
- **RAM**: ≥ 4 GiB
- **Disk**: ≥ 2 GiB free
- **OS**: Windows 10/11 (PowerShell 5.1+) or Linux (`bash`, `sha256sum`)

## Divergence policy

- The kernel binary size is **byte-stable** across reproductions on the
  same toolchain. Acceptable variation: 0 bytes.
- The audit log SHA-256 is **content-stable** across reproductions
  modulo timing — only the `time:` lines may vary. Test `pass` counts
  must match exactly.
- A reproduction whose `test_summary_lines` differs in pass-count from
  the committed baseline is a HARD FAIL. CI gate
  `P0.4 — external reproducibility` enforces this.

## Banned terminology

The reproduction harness DOES NOT validate documentation; that gate
lives in `tests/performance_claim_integrity.rs`. Banned headline terms
(see `crates/qratum-arbiter/src/performance_truth.rs::BANNED_TERMS`)
must not appear in `README.md`, `spec/**`, or any committed report.

## Failure conditions

- Toolchain mismatch → non-zero exit, `bootstrap` step.
- Kernel build failure → non-zero exit, step 1.
- Any test failure (`--no-fail-fast`) → non-zero exit, step 3.
- Hash mismatch vs CI baseline → non-zero exit in CI workflow.
