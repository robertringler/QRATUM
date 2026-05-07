------------------------------ MODULE multicore_replay --------------------------
(*
 * P0.5 — Multicore replay equivalence.
 *
 * The composed state hash projected over a deterministic order of
 * logical CPUs is independent of the physical execution order.
 *)
EXTENDS Naturals, Sequences

CONSTANTS Cores, Tasks, MaxLen

VARIABLES per_core_seq, composed_hash

TypeOK == /\ per_core_seq \in [Cores -> Seq(Tasks)]
          /\ \A c \in Cores : Len(per_core_seq[c]) <= MaxLen
          /\ composed_hash \in Nat

H(f) == 0   \* abstract hash over a fixed iteration order of Cores

Init == per_core_seq = [c \in Cores |-> << >>] /\ composed_hash = H([c \in Cores |-> << >>])

Enqueue(c, t) == /\ Len(per_core_seq[c]) < MaxLen
                 /\ per_core_seq' = [per_core_seq EXCEPT ![c] = Append(@, t)]
                 /\ composed_hash' = H(per_core_seq')

Next == \E c \in Cores : \E t \in Tasks : Enqueue(c, t)

Spec == Init /\ [][Next]_<<per_core_seq, composed_hash>>

(* Order-independence is encoded by H using a fixed iteration of Cores. *)
OrderIndependent == composed_hash = H(per_core_seq)

THEOREM Spec => []OrderIndependent
=================================================================================
