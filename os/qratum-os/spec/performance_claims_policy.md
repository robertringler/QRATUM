# QRATUM Performance Claims Policy

**Status**: P0.5 — verified, enforced by `claim_language_enforcement` test
gate.

This document defines the language and provenance requirements for every
quantitative performance claim emitted by QRATUM CORE OS or any of its
companion crates.

---

## 1. Approved terminology

The following are the **only** approved terms for describing QRATUM's
measured execution-fabric effects:

- scheduler event reduction
- canonical execution reduction
- replay-stable execution compression
- semantic-preserving execution reduction

Every claim must use one of the four phrases above as its kernel verb. Any
other phrasing is rejected by the language-enforcement test.

## 2. Required provenance fields

A performance claim is a structured record. It SHALL include all of the
following fields:

| Field            | Description                                                |
|------------------|------------------------------------------------------------|
| workload type    | Approved workload class identifier (e.g. `arbitration_burst`) |
| duplication ratio| Fraction of duplicated canonical keys, ∈ [0, 1]           |
| core count       | Number of online logical CPUs                              |
| NUMA topology    | `flat`, `dual`, or vendor-specified topology id            |
| IRQ rate         | Interrupts per second observed during measurement          |
| cache state      | `cold`, `warm`, `hot`                                      |
| replay mode      | `live`, `replay`, `golden`                                 |
| timing source    | `rdtsc`, `monotonic_ns`                                    |
| hardware profile | A `HardwareProfile` record from `qratum-arbiter`           |

A claim missing any field is rejected by `claim_registry_validation_tests`
and by CI gate **P0.5 — claim language enforcement**.

## 3. Prohibited terminology

The following terms are **forbidden** in every artifact under this
repository, including reports, README files, marketing copy, source
comments, and documentation, EXCEPT this single listing in this file:

- universal acceleration
- CPU acceleration
- operating-system-wide speedup
- general compute multiplier

## 4. Enforcement

The test `claim_language_enforcement` walks this file and the registered
claim set. Any forbidden term occurring more than once across the
repository, or any approved term missing from this policy, fails CI.
