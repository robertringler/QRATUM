---- MODULE replay_checkpoint_recovery ----
EXTENDS Naturals, Sequences, TLC

(* P0.6 INV-U: A checkpoint saved at epoch e and later restored
   reproduces the merkle root of the state at e. *)

CONSTANTS MaxEpoch
VARIABLES live_root, ckpt_root, restored_root, epoch, saved

Init ==
  /\ live_root     = 0
  /\ ckpt_root     = 0
  /\ restored_root = 0
  /\ epoch         = 0
  /\ saved         = FALSE

Append ==
  /\ epoch < MaxEpoch
  /\ live_root' = live_root + 1
  /\ epoch' = epoch + 1
  /\ UNCHANGED <<ckpt_root, restored_root, saved>>

Save ==
  /\ ~saved
  /\ ckpt_root' = live_root
  /\ saved' = TRUE
  /\ UNCHANGED <<live_root, restored_root, epoch>>

Restore ==
  /\ saved
  /\ restored_root' = ckpt_root
  /\ UNCHANGED <<live_root, ckpt_root, epoch, saved>>

Next == Append \/ Save \/ Restore

Spec == Init /\ [][Next]_<<live_root, ckpt_root, restored_root, epoch, saved>>

INV_U == saved => (restored_root = 0 \/ restored_root = ckpt_root)
====
