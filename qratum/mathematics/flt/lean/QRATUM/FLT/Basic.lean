import Mathlib.NumberTheory.FLT.Basic

namespace QRATUM.FLT

/-- QRATUM alias for Mathlib's canonical fixed-exponent FLT statement. -/
abbrev FermatAt (n : ℕ) : Prop := _root_.FermatLastTheoremFor n

/-- QRATUM alias for Mathlib's canonical full FLT statement. -/
abbrev Statement : Prop := _root_.FermatLastTheorem

/-- FLT at an exponent lifts to every multiple of that exponent. This theorem is
not reproved locally; QRATUM exposes and audits Mathlib's kernel-checked result. -/
theorem fermatAt_of_dvd {d n : ℕ} (hdn : d ∣ n) (h : FermatAt d) : FermatAt n :=
  FermatLastTheoremFor.mono hdn h

end QRATUM.FLT
