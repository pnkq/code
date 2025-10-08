import Mathlib

section
variable (R: Type*) [Ring R]

#check (add_assoc : ∀ a b c : R, a + b + c = a + (b + c))
#check (add_comm : ∀ a b : R, a + b = b + a)
#check (zero_add : ∀ a : R, 0 + a = a)
#check (neg_add_cancel : ∀ a : R, -a + a = 0)
#check (mul_assoc : ∀ a b c : R, a * b * c = a * (b * c))
#check (mul_one : ∀ a : R, a * 1 = a)
#check (one_mul : ∀ a : R, 1 * a = a)
#check (mul_add : ∀ a b c : R, a * (b + c) = a * b + a * c)
#check (add_mul : ∀ a b c : R, (a + b) * c = a * c + b * c)
end


section
variable (CR : Type*) [CommRing CR]
variable (a b c d : CR)

example : c * b * a = b * (a * c) := by ring
end

namespace MyRing
variable {R : Type*} [Ring R]

theorem add_zero (a : R) : a + 0 = a := by rw [add_comm, zero_add]

theorem add_neg_cancel (a : R) : a + -a = 0 := by rw [add_comm, neg_add_cancel]

theorem neg_add_cancel_left (a b : R) : -a + (a + b) = b := by
  rw [← add_assoc, neg_add_cancel, zero_add]

theorem add_neg_cancel_right (a b : R) : a + b + -b = a := by
  rw [add_assoc, add_neg_cancel, add_zero]

theorem add_left_cancel {a b c : R} (h : a + b = a + c) : b = c := by
  rw [← neg_add_cancel_left a b] -- b = -a + (a + b)
  rw [h] -- => -a + (a + c)
  rw [← add_assoc] -- => -a + a +
  rw [neg_add_cancel] -- => 0 + c
  rw [zero_add]

theorem add_right_cancel {a b c : R} (h : a + b = c + b) : a = c := by
  rw [← add_neg_cancel_right a b] -- a = a + b + -b
  rw [h] -- => c + b + -b
  rw [add_assoc] -- => c + (b + -b)
  rw [add_neg_cancel] -- => c + 0
  rw [add_zero]

-- #check MyRing.add_zero
-- #check add_zero

theorem mul_zero (a : R) : a * 0 = 0 := by
  have h: a * 0 + a * 0 = a * 0 + 0 := by -- introduce a new hypothesis
    rw [← mul_add, add_zero, add_zero] -- => a * (0 + 0) => a * 0 => a * 0 + 0
  rw [add_left_cancel h]


theorem zero_mul (a : R) : 0 * a = 0 := by
  have h: 0 * a + 0 * a = 0 * a + 0 := by
    rw [← add_mul, add_zero, add_zero] -- => (0 + 0) * a => 0 * a
  rw [add_left_cancel h]

-- phuonglh 1
theorem neg_eq_of_add_eq_zero {a b : R} (h : a + b = 0) : -a = b := by
  rw [← add_zero (-a), ← h, neg_add_cancel_left] -- => -a + 0 => -a + (a + b) => b
-- phuonglh 2
theorem neg_eq_of_add_eq_zero_2 {a b : R} (h : a + b = 0) : -a = b := by
  rw [← neg_add_cancel_left a b, h, add_zero] -- => -a + (a + b) => -a + 0 => -a

-- phuonglh 1
theorem eq_neg_of_add_eq_zero {a b : R} (h : a + b = 0) : a = -b := by
  rw [← add_neg_cancel_right a b, h, zero_add] -- => (a + b) + -b => 0 + -b => -b

theorem eq_neg_of_add_eq_zero_2 {a b : R} (h : a + b = 0) : a = -b := by
  symm -- -b = a
  apply neg_eq_of_add_eq_zero -- => a = -b
  rw [add_comm, h] -- ??


theorem neg_zero : (-0 : R) = 0 := by
  apply neg_eq_of_add_eq_zero
  rw [add_zero]

theorem neg_neg (a : R) : - -a = a := by
  apply neg_eq_of_add_eq_zero
  rw [neg_add_cancel]


theorem self_sub (a : R) : a - a = 0 := by
  rw [sub_eq_add_neg, add_neg_cancel] -- => a + -a => 0

theorem one_add_one_eq_two : 1 + 1 = (2 : R) := by
  norm_num

theorem two_mul (a : R) : 2 * a = a + a := by
  rw [← one_add_one_eq_two, add_mul, one_mul]

end MyRing

namespace MyGroup

variable {G : Type*} [Group G]

#check (mul_assoc : ∀ a b c : G, a * b * c = a * (b * c))
#check (one_mul : ∀ a : G, 1 * a = a)
#check (inv_mul_cancel : ∀ a : G, a⁻¹ * a = 1)

-- theorem mul_inv_cancel (a : G) : a * a⁻¹ = 1 := by
--   sorry

-- theorem mul_one (a : G) : a * 1 = a := by
--   sorry

-- theorem mul_inv_rev (a b : G) : (a * b)⁻¹ = b⁻¹ * a⁻¹ := by
--   sorry

end MyGroup
