import Mathlib.NumberTheory.FLT.Four
import QRATUM.FLT.Basic

namespace QRATUM.FLT

/-- Every exponent greater than two is divisible by four or has an odd prime
factor. QRATUM exposes Mathlib's proved arithmetic decomposition. -/
theorem exponent_coverage {n : ℕ} (hn : 2 < n) :
    4 ∣ n ∨ ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ Odd p :=
  Nat.four_dvd_or_exists_odd_prime_and_dvd_of_two_lt hn

/-- The exponent-four case is already formalized in Mathlib by infinite descent. -/
theorem exponent_four : FermatAt 4 :=
  fermatLastTheoremFour

/-- M0: proving FLT for odd prime exponents suffices for the full theorem.
Mathlib supplies both the exponent-four case and the arithmetic reduction. -/
theorem flt_of_odd_primes
    (hOddPrime : ∀ p : ℕ, Nat.Prime p → Odd p → FermatAt p) : Statement :=
  FermatLastTheorem.of_odd_primes hOddPrime

end QRATUM.FLT
