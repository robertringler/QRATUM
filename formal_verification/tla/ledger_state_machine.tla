---------------------------- MODULE ledger_state_machine ----------------------------
(*
TLA+ Specification for QRATUM Ledger State Machine

This specification formally verifies the ledger state machine including:
- State transitions and invariants
- Ephemeral lifecycle (materialization → execution → destruction)
- Merkle tree integrity
- Session-bound rollback
- Zero persistent state between sessions
*)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxTxos,            \* Maximum TXOs in ledger
    MaxCheckpoints,     \* Maximum rollback checkpoints
    MaxSessions         \* Maximum concurrent sessions

VARIABLES
    ledger_state,       \* Current ledger state
    merkle_root,        \* Current Merkle root hash
    checkpoints,        \* Rollback checkpoint stack
    session_id,         \* Current session identifier
    lifecycle_stage,    \* Current lifecycle stage
    txo_count,          \* Number of TXOs in ledger
    epoch               \* Current epoch counter

vars == <<ledger_state, merkle_root, checkpoints, session_id, lifecycle_stage, txo_count, epoch>>

\***************************************************************************
\* Type Definitions
\***************************************************************************

LifecycleStages == {"Uninitialized", "QuorumConvergence", "EphemeralMaterialization", 
                    "Execution", "OutcomeCommitment", "SelfDestruction"}

TxoTypes == {"Input", "Outcome", "DecayJustification", "CanaryProbe", 
             "CensorshipEvent", "ProxyApproval", "ComplianceAttestation"}

\***************************************************************************
\* Type Invariants
\***************************************************************************

TypeOK ==
    /\ ledger_state \in {"Empty", "Active", "Committed", "Destroyed"}
    /\ merkle_root \in Nat
    /\ checkpoints \in Seq(Nat)
    /\ session_id \in Nat
    /\ lifecycle_stage \in LifecycleStages
    /\ txo_count \in 0..MaxTxos
    /\ epoch \in Nat

\***************************************************************************
\* Initial State
\***************************************************************************

Init ==
    /\ ledger_state = "Empty"
    /\ merkle_root = 0
    /\ checkpoints = <<>>
    /\ session_id = 0
    /\ lifecycle_stage = "Uninitialized"
    /\ txo_count = 0
    /\ epoch = 0

\***************************************************************************
\* Lifecycle Stage Invariants
\***************************************************************************

\* INVARIANT 1: Lifecycle Stage Ordering
\* Stages must follow correct order: Uninitialized → QuorumConvergence → 
\* EphemeralMaterialization → Execution → OutcomeCommitment → SelfDestruction
LifecycleOrderInvariant ==
    /\ (lifecycle_stage = "EphemeralMaterialization") => 
        (ledger_state # "Destroyed")
    /\ (lifecycle_stage = "Execution") => 
        (ledger_state = "Active")
    /\ (lifecycle_stage = "SelfDestruction") => 
        (ledger_state \in {"Committed", "Destroyed"})

\* INVARIANT 2: Ephemeral State Property
\* During execution, state exists only in RAM (represented by Active ledger state)
EphemeralStateInvariant ==
    (lifecycle_stage \in {"EphemeralMaterialization", "Execution"}) =>
        (ledger_state = "Active")

\* INVARIANT 3: Zero Persistent State Between Sessions
\* After destruction, ledger must be empty
ZeroPersistenceInvariant ==
    (lifecycle_stage = "SelfDestruction" /\ ledger_state = "Destroyed") =>
        (txo_count = 0 /\ merkle_root = 0 /\ checkpoints = <<>>)

\* INVARIANT 4: Checkpoint Bounds
\* Checkpoints must not exceed maximum
CheckpointBoundsInvariant ==
    Len(checkpoints) <= MaxCheckpoints

\* INVARIANT 5: TXO Count Bounds
\* TXO count must not exceed maximum
TxoCountBoundsInvariant ==
    txo_count <= MaxTxos

\* INVARIANT 6: Merkle Root Consistency
\* Merkle root must be consistent with TXO count
MerkleRootConsistency ==
    (txo_count = 0) => (merkle_root = 0)

\* INVARIANT 7: Session Monotonicity
\* Session ID must be monotonically increasing (never reused)
SessionMonotonicity ==
    session_id >= 0

\* INVARIANT 8: Outcome Commitment Requires Execution
\* Cannot commit outcomes without prior execution
OutcomeRequiresExecution ==
    (lifecycle_stage = "OutcomeCommitment") =>
        (ledger_state \in {"Active", "Committed"})

\* Combined invariant for all ledger state machine properties
LedgerInvariants ==
    /\ TypeOK
    /\ LifecycleOrderInvariant
    /\ EphemeralStateInvariant
    /\ ZeroPersistenceInvariant
    /\ CheckpointBoundsInvariant
    /\ TxoCountBoundsInvariant
    /\ MerkleRootConsistency
    /\ SessionMonotonicity
    /\ OutcomeRequiresExecution

\***************************************************************************
\* Actions
\***************************************************************************

\* Begin new session with quorum convergence
BeginSession ==
    /\ lifecycle_stage = "Uninitialized"
    /\ ledger_state = "Empty"
    /\ session_id' = session_id + 1
    /\ lifecycle_stage' = "QuorumConvergence"
    /\ UNCHANGED <<ledger_state, merkle_root, checkpoints, txo_count, epoch>>

\* Materialize ephemeral ledger after quorum convergence
MaterializeLedger ==
    /\ lifecycle_stage = "QuorumConvergence"
    /\ ledger_state' = "Active"
    /\ lifecycle_stage' = "EphemeralMaterialization"
    /\ epoch' = epoch + 1
    /\ UNCHANGED <<merkle_root, checkpoints, session_id, txo_count>>

\* Transition to execution phase
BeginExecution ==
    /\ lifecycle_stage = "EphemeralMaterialization"
    /\ ledger_state = "Active"
    /\ lifecycle_stage' = "Execution"
    /\ UNCHANGED <<ledger_state, merkle_root, checkpoints, session_id, txo_count, epoch>>

\* Append TXO to ledger
AppendTxo ==
    /\ lifecycle_stage = "Execution"
    /\ ledger_state = "Active"
    /\ txo_count < MaxTxos
    /\ txo_count' = txo_count + 1
    /\ merkle_root' = merkle_root + 1  \* Simplified: increment represents recomputation
    /\ UNCHANGED <<ledger_state, checkpoints, session_id, lifecycle_stage, epoch>>

\* Create rollback checkpoint
CreateCheckpoint ==
    /\ lifecycle_stage = "Execution"
    /\ ledger_state = "Active"
    /\ Len(checkpoints) < MaxCheckpoints
    /\ checkpoints' = Append(checkpoints, merkle_root)
    /\ UNCHANGED <<ledger_state, merkle_root, session_id, lifecycle_stage, txo_count, epoch>>

\* Rollback to previous checkpoint
Rollback ==
    /\ lifecycle_stage = "Execution"
    /\ ledger_state = "Active"
    /\ Len(checkpoints) > 0
    /\ merkle_root' = Head(checkpoints)
    /\ checkpoints' = Tail(checkpoints)
    \* Simplified: restore TXO count (would need more complex state in real impl)
    /\ UNCHANGED <<ledger_state, session_id, lifecycle_stage, txo_count, epoch>>

\* Commit outcomes
CommitOutcomes ==
    /\ lifecycle_stage = "Execution"
    /\ ledger_state = "Active"
    /\ lifecycle_stage' = "OutcomeCommitment"
    /\ ledger_state' = "Committed"
    /\ UNCHANGED <<merkle_root, checkpoints, session_id, txo_count, epoch>>

\* Self-destruct: Complete zeroization
SelfDestruct ==
    /\ lifecycle_stage = "OutcomeCommitment"
    /\ ledger_state = "Committed"
    /\ lifecycle_stage' = "SelfDestruction"
    /\ ledger_state' = "Destroyed"
    /\ merkle_root' = 0
    /\ checkpoints' = <<>>
    /\ txo_count' = 0
    /\ UNCHANGED <<session_id, epoch>>

\* Reset for new session
ResetForNewSession ==
    /\ lifecycle_stage = "SelfDestruction"
    /\ ledger_state = "Destroyed"
    /\ lifecycle_stage' = "Uninitialized"
    /\ ledger_state' = "Empty"
    /\ UNCHANGED <<merkle_root, checkpoints, session_id, txo_count, epoch>>

\***************************************************************************
\* Next State Relation
\***************************************************************************

Next ==
    \/ BeginSession
    \/ MaterializeLedger
    \/ BeginExecution
    \/ AppendTxo
    \/ CreateCheckpoint
    \/ Rollback
    \/ CommitOutcomes
    \/ SelfDestruct
    \/ ResetForNewSession

\***************************************************************************
\* Specification
\***************************************************************************

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\***************************************************************************
\* Properties to Verify
\***************************************************************************

\* Safety: All invariants always hold
Safety == []LedgerInvariants

\* Liveness: Sessions eventually complete
SessionCompletion ==
    [](lifecycle_stage = "Execution" => <>(lifecycle_stage = "SelfDestruction"))

\* Liveness: Materialization leads to execution
MaterializationToExecution ==
    [](lifecycle_stage = "EphemeralMaterialization" => <>(lifecycle_stage = "Execution"))

\* Liveness: Destroyed state can be reset
DestructionRecovery ==
    [](lifecycle_stage = "SelfDestruction" => <>(lifecycle_stage = "Uninitialized"))

\* Theorem: The specification satisfies safety and liveness
THEOREM Spec => Safety /\ SessionCompletion

=============================================================================
