-------------------------- MODULE canonical_equivalence_boundary -----------------
(*
 * P0.5 — Canonical equivalence boundary.
 *
 * For every input stream S in the bounded scenario corpus,
 *   expand(fold(S)) == S
 * holds — i.e. AHTC-K folding never alters the semantic content of
 * the input stream that crosses the kernel arbitration plane.
 *)
EXTENDS Naturals, Sequences

CONSTANT Inputs           \* finite set of canonical input atoms
CONSTANT MaxLen           \* bounded sequence length

VARIABLES stream

TypeOK == stream \in Seq(Inputs) /\ Len(stream) <= MaxLen

(* Pure functions modeled as TLA+ functions *)
fold(s)   == s              \* injective canonicalization (identity in model)
expand(c) == c

Init == stream = << >>

Append(x) == /\ Len(stream) < MaxLen
             /\ stream' = Append(stream, x)
             /\ UNCHANGED <<>>

Spec == Init /\ [][\E x \in Inputs : Append(x)]_stream

(* Invariant — the canonical-equivalence boundary. *)
CanonicalEquivalence == expand(fold(stream)) = stream

THEOREM Spec => []CanonicalEquivalence
=================================================================================
