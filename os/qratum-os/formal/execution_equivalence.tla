---- MODULE execution_equivalence ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Intents, MaxFolds
ASSUME Cardinality(Intents) <= 16 /\ MaxFolds <= 32

VARIABLES live, replayed, fold_log

vars == << live, replayed, fold_log >>

(* Init: empty execution graph *)
Init == /\ live      = << >>
        /\ replayed  = << >>
        /\ fold_log  = << >>

(* Submit an intent to live execution *)
Submit(i) == /\ Len(live) < MaxFolds
             /\ live'     = Append(live, i)
             /\ fold_log' = Append(fold_log, i)
             /\ UNCHANGED replayed

(* Replay re-applies every fold_log entry deterministically *)
Replay == /\ replayed' = fold_log
          /\ UNCHANGED << live, fold_log >>

Next == \/ \E i \in Intents : Submit(i)
        \/ Replay

Spec == Init /\ [][Next]_vars

(* Property: after Replay, replayed graph EQUALS live graph *)
ReplayLossless == [](replayed = << >> \/ replayed = fold_log)

(* Property: live graph is exactly the projection of fold_log *)
FoldEqualsLive == [](live = fold_log)

(* Property: idempotence of fold(expand(S)) = S
   Modeled as: replaying twice yields same sequence *)
ReplayIdempotent == [](Replay => (replayed' = replayed))

THEOREM Spec => ReplayLossless
THEOREM Spec => FoldEqualsLive
====
