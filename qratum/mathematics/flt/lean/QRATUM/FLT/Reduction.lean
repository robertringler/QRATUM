import QRATUM.FLT.Basic

namespace QRATUM.FLT

/-- Arithmetic coverage needed by the M0 reduction: every exponent greater than
2 is divisible by 4 or has an odd prime divisor. This is isolated so the
number-theoretic decomposition can be proved and audited independently. -/
def ExponentCoverage : Prop :=
  ∀ n : ℕ, 2 < n → 4 ∣ n ∨ ∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ p ∣ n

/-- M0 assembly theorem. Given exponent coverage, FLT at exponent 4, and FLT
at every odd prime exponent, FLT follows at every exponent greater than 2. -/
theorem flt_of_four_and_odd_primes
    (coverage : ExponentCoverage)
    (hFour : FermatAt 4)
    (hOddPrime : ∀ p : ℕ, Nat.Prime p → Odd p → FermatAt p) :
    FermatLastTheorem := by
  intro n hn
  rcases coverage n hn with hFourDivides | ⟨p, hp, hOdd, hpDivides⟩
  · exact fermatAt_of_dvd (by norm_num) (by omega) hFourDivides hFour
  · exact fermatAt_of_dvd hp.pos (by omega) hpDivides (hOddPrime p hp hOdd)

end QRATUM.FLT
