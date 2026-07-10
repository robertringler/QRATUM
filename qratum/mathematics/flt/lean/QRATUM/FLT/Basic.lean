import Mathlib

namespace QRATUM.FLT

/-- Fermat's equation has no positive natural-number solution at exponent `n`. -/
def FermatAt (n : ℕ) : Prop :=
  ∀ x y z : ℕ, 0 < x → 0 < y → 0 < z → x ^ n + y ^ n ≠ z ^ n

/-- Full Fermat's Last Theorem over positive natural numbers. -/
def FermatLastTheorem : Prop :=
  ∀ n : ℕ, 2 < n → FermatAt n

/-- A counterexample at exponent `n` descends to any positive divisor `d` of
`n`, by replacing each base with its `n / d`-th power. Equivalently, FLT at
`d` lifts to every positive multiple `n` of `d`. -/
theorem fermatAt_of_dvd
    {d n : ℕ}
    (hd : 0 < d)
    (hn : 0 < n)
    (hdn : d ∣ n)
    (h : FermatAt d) : FermatAt n := by
  rcases hdn with ⟨k, rfl⟩
  have hk : 0 < k := by omega
  intro x y z hx hy hz hxyz
  apply h (x ^ k) (y ^ k) (z ^ k) (by positivity) (by positivity) (by positivity)
  simpa [pow_mul, Nat.mul_comm] using hxyz

end QRATUM.FLT
