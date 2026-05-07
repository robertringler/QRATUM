---- MODULE hardware_replay_consistency ----
EXTENDS Naturals, Sequences, TLC

(* P0.6: live capture and replay of a hardware event stream produce
   identical stream hashes. The hash is modeled as a fold of any
   commutative-injection function H. *)

CONSTANTS H(_,_), Events
VARIABLES live, replay

Init ==
  /\ live   = << >>
  /\ replay = << >>

Capture(e) == live' = Append(live, e) /\ UNCHANGED replay
Replay     == replay' = live /\ UNCHANGED live

StreamHash(s) ==
  LET RECURSIVE F(_,_)
      F(seed, t) == IF Len(t) = 0 THEN seed ELSE F(H(seed, t[1]), Tail(t))
  IN F(0, s)

Spec == Init /\ [][\E e \in Events: Capture(e) \/ Replay]_<<live, replay>>

INV_REPLAY == StreamHash(live) = StreamHash(replay) \/ replay = << >>
====
