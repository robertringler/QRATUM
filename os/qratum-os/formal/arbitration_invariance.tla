---- MODULE arbitration_invariance ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANT Intents
ASSUME Cardinality(Intents) <= 16

VARIABLES raw_stream, folded_stream, verdict_raw, verdict_folded

vars == << raw_stream, folded_stream, verdict_raw, verdict_folded >>

(* Abstract verdict: depends ONLY on canonical key, never on order
   or duplication. *)
Verdict(stream) == { i \in Intents : \E k \in 1..Len(stream) : stream[k] = i }

Init == /\ raw_stream      = << >>
        /\ folded_stream   = << >>
        /\ verdict_raw     = {}
        /\ verdict_folded  = {}

Submit(i) == /\ raw_stream'    = Append(raw_stream, i)
             /\ folded_stream' = IF \E k \in 1..Len(folded_stream): folded_stream[k] = i
                                 THEN folded_stream
                                 ELSE Append(folded_stream, i)
             /\ verdict_raw'   = Verdict(raw_stream')
             /\ verdict_folded'= Verdict(folded_stream')

Next == \E i \in Intents : Submit(i)

Spec == Init /\ [][Next]_vars

(* Arbitration invariance: the verdict is identical on raw and
   folded streams. INV-N enforces no false fold; this property
   asserts the inverse — folding NEVER changes arbitration result. *)
ArbitrationInvariant == [](verdict_raw = verdict_folded)

THEOREM Spec => ArbitrationInvariant
====
