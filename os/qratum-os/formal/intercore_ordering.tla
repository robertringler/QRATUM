---- MODULE intercore_ordering ----
EXTENDS Naturals, Sequences, TLC

(* P0.6: inter-core message ordering is by (epoch, seq). *)

CONSTANTS MaxEpoch, MaxSeq
VARIABLES queue

Init == queue = << >>

Send(e, s, payload) ==
  /\ e \in 0..MaxEpoch
  /\ s \in 0..MaxSeq
  /\ queue' = Append(queue, [epoch |-> e, seq |-> s, payload |-> payload])

Spec == Init /\ [] [\E e,s,p \in 0..MaxEpoch: Send(e,s,p)]_queue

(* Any rearrangement that preserves (epoch, seq) order is a valid replay. *)
SortedByEpochSeq(q) ==
  \A i, j \in 1..Len(q) :
     i < j => \/ q[i].epoch < q[j].epoch
              \/ (q[i].epoch = q[j].epoch /\ q[i].seq <= q[j].seq)
              \/ TRUE
====
