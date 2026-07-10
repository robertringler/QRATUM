# QRATUM-FLT

QRATUM-FLT is a formal-verification and proof-audit program for Fermat's Last
Theorem (FLT). It does **not** claim to rediscover the theorem proved by Andrew
Wiles and Richard Taylor.

## Milestone M0 — exponent reduction

M0 establishes:

> If FLT holds for every odd prime exponent, then FLT holds for every natural
> exponent greater than two.

The live Mathlib dependency already contains kernel-checked proofs of:

- `FermatLastTheoremWith.mono`: FLT at an exponent lifts to its multiples;
- `Nat.four_dvd_or_exists_odd_prime_and_dvd_of_two_lt`: every exponent above
  two is divisible by four or has an odd prime divisor;
- `fermatLastTheoremFour`: the exponent-four case by infinite descent;
- `FermatLastTheorem.of_odd_primes`: the complete M0 reduction.

QRATUM therefore integrates and audits these canonical results instead of
maintaining a duplicate provisional proof.

## Current frontier

After M0, the unresolved formalization target is the odd-prime theorem family:

```lean
∀ p : ℕ, Nat.Prime p → Odd p → FermatLastTheoremFor p
```

Completing that target requires the deep Frey–Ribet–Taylor–Wiles chain or a
verified equivalent. QRATUM must treat every imported result and remaining gap
as an explicit dependency.

## Trust boundary

A QRATUM milestone is accepted only when:

- Lean compiles the target theorem;
- no `sorry`, `admit`, local axiom, or unsafe theorem remains;
- imported results are pinned and traced to their source modules;
- a clean build reproduces from the pinned toolchain and dependency revision.

## Layout

- `lean/QRATUM/FLT/Basic.lean`: aliases and audited wrappers around Mathlib's
  canonical FLT API.
- `lean/QRATUM/FLT/Reduction.lean`: M0 integration.
- `blueprint/dependency_graph.json`: machine-readable proof graph.
- `scripts/audit_placeholders.py`: placeholder and local-axiom rejection.

## Build

```bash
cd qratum/mathematics/flt
lake update
lake build
python scripts/audit_placeholders.py
```

## Honest status

The mathematical content of M0 exists in Mathlib and is now wired into QRATUM.
QRATUM's own wrapper package remains **build-pending** until its dedicated CI
job completes successfully. The full FLT proof is not complete because the
odd-prime theorem family remains the central deep dependency.
