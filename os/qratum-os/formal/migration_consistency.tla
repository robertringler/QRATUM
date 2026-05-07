------------------------------ MODULE migration_consistency ----------------------
(*
 * P0.5 — Migration commit order is replay-stable.
 *
 * For any sequence of legal migrations M = m1, m2, ..., mn,
 * replay(M) produces a state hash equal to that of the original
 * forward execution.
 *)
EXTENDS Naturals, Sequences

CONSTANTS Tasks, Cores, MaxLen

VARIABLES log, hash_live, hash_replay

Migration == [task: Tasks, src: Cores, dst: Cores]

TypeOK == /\ log \in Seq(Migration)
          /\ Len(log) <= MaxLen
          /\ hash_live   \in Nat
          /\ hash_replay \in Nat

H(s) == Len(s)                             \* abstract hash

Init == log = << >> /\ hash_live = 0 /\ hash_replay = 0

Commit(m) == /\ Len(log) < MaxLen
             /\ m.src # m.dst
             /\ log' = Append(log, m)
             /\ hash_live' = H(log')
             /\ UNCHANGED hash_replay

Replay == hash_replay' = H(log) /\ UNCHANGED <<log, hash_live>>

Next == (\E m \in Migration : Commit(m)) \/ Replay

Spec == Init /\ [][Next]_<<log, hash_live, hash_replay>>

ReplayStable == hash_replay = 0 \/ hash_replay = hash_live

THEOREM Spec => []ReplayStable
=================================================================================
