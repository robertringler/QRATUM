---- MODULE replay_consistency ----
EXTENDS Naturals, Sequences

CONSTANTS Tasks, Cores
ASSUME Cardinality(Tasks) <= 16 /\ Cardinality(Cores) <= 4

VARIABLES mig_log_a, mig_log_b, sched_hash_a, sched_hash_b

vars == << mig_log_a, mig_log_b, sched_hash_a, sched_hash_b >>

(* Migration frame *)
Frame(t, src, dst, seq) == [task |-> t, src |-> src, dst |-> dst, seq |-> seq]

(* Pure function from migration log to scheduler hash *)
SchedulerHash(log) == log    \* abstract: any deterministic function

Init == /\ mig_log_a    = << >>
        /\ mig_log_b    = << >>
        /\ sched_hash_a = SchedulerHash(<< >>)
        /\ sched_hash_b = SchedulerHash(<< >>)

(* Apply identical migration to both runs *)
Migrate(t, src, dst) ==
  /\ src # dst
  /\ LET seq == Len(mig_log_a) + 1 IN
       /\ mig_log_a'    = Append(mig_log_a, Frame(t, src, dst, seq))
       /\ mig_log_b'    = Append(mig_log_b, Frame(t, src, dst, seq))
       /\ sched_hash_a' = SchedulerHash(mig_log_a')
       /\ sched_hash_b' = SchedulerHash(mig_log_b')

Next == \E t \in Tasks, s \in Cores, d \in Cores : Migrate(t, s, d)

Spec == Init /\ [][Next]_vars

(* INV-L: identical migration log → identical scheduler hash *)
ReplayConsistency == [](mig_log_a = mig_log_b => sched_hash_a = sched_hash_b)

(* No orphaned execution states — every migrated task ends up on
   exactly one queue (modeled as: every Frame has src # dst). *)
NoOrphanedStates == [](\A k \in 1..Len(mig_log_a) : mig_log_a[k].src # mig_log_a[k].dst)

THEOREM Spec => ReplayConsistency
THEOREM Spec => NoOrphanedStates
====
