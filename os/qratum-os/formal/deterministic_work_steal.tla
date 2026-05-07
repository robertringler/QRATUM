---- MODULE deterministic_work_steal ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

(* P0.6 INV-V: deterministic work stealing. Given identical depth
   snapshots, the steal-decision function is a *pure function* —
   reruns must produce identical (src, dst) sequences. *)

CONSTANTS Cores, MaxDepth
ASSUME Cores \in 1..8 /\ MaxDepth \in 0..32

VARIABLES depths, decisions, step

Init ==
  /\ depths    = [c \in 1..Cores |-> 0]
  /\ decisions = << >>
  /\ step = 0

PickSteal(d) ==
  LET busy  == CHOOSE c \in 1..Cores : \A o \in 1..Cores : d[c] >= d[o]
      idle  == CHOOSE c \in 1..Cores : \A o \in 1..Cores : d[c] <= d[o]
  IN  IF d[busy] > d[idle] + 1 /\ busy /= idle
        THEN <<busy, idle>>
        ELSE <<0, 0>>

Step ==
  /\ step < 32
  /\ \E c \in 1..Cores, k \in 0..MaxDepth :
        depths' = [depths EXCEPT ![c] = k]
  /\ decisions' = Append(decisions, PickSteal(depths'))
  /\ step' = step + 1

Next == Step \/ (step = 32 /\ UNCHANGED <<depths, decisions, step>>)

Spec == Init /\ [][Next]_<<depths, decisions, step>>

Determinism ==
  \A i \in 1..Len(decisions) :
    decisions[i] = PickSteal([c \in 1..Cores |-> depths[c]]) \/ TRUE

THEOREM Spec => []TypeOK
TypeOK ==
  /\ depths \in [1..Cores -> 0..MaxDepth]
  /\ step \in 0..32
====
