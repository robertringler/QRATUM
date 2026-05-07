---- MODULE divergence_envelope ----
EXTENDS Naturals, TLC

(* P0.6 INV-Y: An "envelope" is a tuple of bound dimensions. The
   replay is non-divergent iff every observed dimension lies inside
   the bound. *)

CONSTANTS BoundJitter, BoundLatency, BoundDrift, BoundPressure
VARIABLES jitter, latency, drift, pressure

Init ==
  /\ jitter   = 0
  /\ latency  = 0
  /\ drift    = 0
  /\ pressure = 0

Observe(j, l, d, p) ==
  /\ jitter'   = j
  /\ latency'  = l
  /\ drift'    = d
  /\ pressure' = p

Spec == Init /\ [] [\E j,l,d,p \in 0..1000000: Observe(j,l,d,p)]_<<jitter,latency,drift,pressure>>

Within ==
  /\ jitter   <= BoundJitter
  /\ latency  <= BoundLatency
  /\ drift    <= BoundDrift
  /\ pressure <= BoundPressure

Divergent == ~Within
====
