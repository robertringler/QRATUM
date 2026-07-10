# QRATUM-FLT

QRATUM-FLT is a formal-verification program for reconstructing and auditing the
modern proof of Fermat's Last Theorem (FLT). It does **not** claim to rediscover
or replace the theorem proved by Andrew Wiles and Richard Taylor.

## Current milestone: M0 — exponent reduction

Formal target:

> If FLT holds for exponent 4 and for every odd prime exponent, then FLT holds
> for every natural exponent greater than 2.

The reduction is elementary but foundational:

1. If an exponent `n > 2` has an odd prime divisor `p`, a solution at exponent
   `n = p * m` induces a solution at exponent `p` by replacing `(x,y,z)` with
   `(x^m,y^m,z^m)`.
2. If `n` has no odd prime divisor, then `n` is a power of two; because `n > 2`,
   it is divisible by 4, and a solution at exponent `n = 4 * m` induces a
   solution at exponent 4.

## Trust boundary

A milestone is accepted only when all of the following hold:

- Lean compiles the target theorem.
- No `sorry`, `admit`, or hidden placeholder remains.
- Any imported mathematical result is stated explicitly as a hypothesis or
  traced to a verified dependency.
- The build is reproducible from pinned toolchain and dependency versions.

## Layout

- `lean/QRATUM/FLT/Basic.lean`: definitions and exponent-lifting lemma.
- `lean/QRATUM/FLT/Reduction.lean`: M0 theorem interface and reduction proof.
- `blueprint/dependency_graph.json`: machine-readable proof graph.
- `scripts/audit_placeholders.py`: rejects common Lean proof placeholders.

## Build

From this directory:

```bash
lake update
lake build
python scripts/audit_placeholders.py
```

## Honest status

The source scaffold and audit tooling are present. Until CI or a local Lean
installation runs `lake build`, compilation is **not yet certified**.
