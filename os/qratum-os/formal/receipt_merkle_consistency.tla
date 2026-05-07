---- MODULE receipt_merkle_consistency ----
EXTENDS Naturals, Sequences, TLC

(* P0.6 INV-X: Merkle root is a pure function of the leaf sequence. *)

CONSTANTS Leaves, Hash(_,_)
VARIABLES tree_a, tree_b

Init ==
  /\ tree_a = << >>
  /\ tree_b = << >>

AppendBoth(leaf) ==
  /\ tree_a' = Append(tree_a, leaf)
  /\ tree_b' = Append(tree_b, leaf)

Root(t) ==
  IF Len(t) = 0 THEN 0
  ELSE LET RECURSIVE Reduce(_)
           Reduce(layer) ==
             IF Len(layer) <= 1 THEN layer[1]
             ELSE Reduce([ i \in 1..((Len(layer)+1) \div 2) |->
                            IF 2*i <= Len(layer)
                              THEN Hash(layer[2*i-1], layer[2*i])
                              ELSE Hash(layer[2*i-1], layer[2*i-1]) ])
       IN Reduce(t)

Spec == Init /\ [] [\E l \in 0..255: AppendBoth(l)]_<<tree_a, tree_b>>

INV_X == Root(tree_a) = Root(tree_b)
====
