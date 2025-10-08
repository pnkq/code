import Mathlib.Data.Real.Basic

example (a b c : ℝ): a * b * c = b * (a * c) := by
  rw [mul_comm a b]
  rw [mul_assoc b a c]

-- first proof (complicated)
example (a b c : ℝ): c * b * a = b * (a * c) := by
  rw [mul_comm c b]
  rw [mul_assoc b c a]
  rw [← mul_comm]
  rw [mul_comm c a]
  rw [mul_comm]

-- second proof (simple)
example (a b c : ℝ): c * b * a = b * (a * c) := by
  rw [mul_comm c b]
  rw [mul_assoc b c a]
  rw [mul_comm c a]

-- first proof
example (a b c : ℝ) : a * (b * c) = b * (a * c) := by
  rw [mul_comm] -- a * (b * c ) = (b * c) * a
  rw [mul_assoc] -- (b * c) * a = b * (c * a)
  rw [mul_comm c a] -- b * (c * a) = b * (a * c)

-- second proof
example (a b c : ℝ) : a * (b * c) = b * (a * c) := by
  rw [← mul_assoc] -- a * (b * c) = a * b * c
  rw [mul_comm a b] -- a * b * c = b * a * c
  rw [mul_assoc] -- b * a * c = b * (a * c)

-- first proof
example (a b c d e f : ℝ) (h : a * b = c * d) (h' : e = f) : a * (b * e) = c * (d * f) := by
  rw [h'] -- a * (b * e) = a * (b * f)
  rw [← mul_assoc] -- => a * b * f
  rw [h] -- => c * d * f
  rw [mul_assoc] -- => c * (d * f)

-- seconf proof
example (a b c d e f : ℝ) (h : a * b = c * d) (h' : e = f) : a * (b * e) = c * (d * f) := by
  rw [← mul_assoc] -- => a * b * e
  rw [h] -- => c * d * e
  rw [h'] -- => c * d * f
  rw [mul_assoc] -- => c * (d * f)


-- first proof
example (a b c d e f : ℝ) (h : b * c = e * f) : a * b * c * d = a * e * f * d := by
  rw [mul_assoc] -- a * b * (c * d)
  rw [mul_assoc] -- a * (b * (c * d))
  rw [← mul_assoc b c d] -- a * (b * c * d)
  rw [h] -- a * ((e * f) * d))
  rw [← mul_assoc] -- a * (e * f) * d
  rw [← mul_assoc] -- a * e * f * d

-- second proof
example (a b c d e f : ℝ) (h : b * c = e * f) : a * b * c * d = a * e * f * d := by
  rw [mul_assoc a] -- a * (b * c * d)
  rw [h] -- a * (e * f * d)
  rw [← mul_assoc]

example (a b c d : ℝ) (hyp : c = b * a - d) (hyp' : d = a * b) : c = 0 := by
  rw [hyp] -- c = b * a - d
  rw [mul_comm] -- a * b - d
  rw [hyp'] -- c = d - d
  rw [sub_self] -- c = 0

example (a b c d e f : ℝ) (h : a * b = c * d) (h' : e = f) : a * (b * e) = c * (d * f) := by
  rw [← mul_assoc, h, mul_assoc, h']

example : (a + b) * (a + b) = a * a + 2 * (a * b) + b * b := by
  rw [mul_add] -- (a+b)*a + (a+b)*b
  rw [add_mul] -- aa + ba + (a+b)b
  rw [add_mul] -- aa + ba + (ab + bb)
  rw [← add_assoc] -- aa + ba + ab + bb
  rw [add_assoc (a * a)] -- aa + (ba + ab) + bb
  rw [mul_comm b a] -- aa + (ab + ab) + bb
  rw [← two_mul] -- aa + 2(ab) + bb

example (a b : ℝ) : (a + b) * (a - b) = a ^ 2 - b ^ 2 := by
  rw [mul_sub] -- (a + b)a - (a + b)b +
  rw [add_mul] -- aa + ba - (a + b)b
  rw [add_mul] -- aa + ba - (ab + bb)
  rw [← sub_sub] -- aa + ba - ab - bb
  rw [mul_comm b a] -- aa + ab - ab - bb
  rw [← add_sub] -- aa + (ab - ab) - bb
  rw [sub_self] -- aa + 0 - bb
  rw [add_zero] -- aa - bb
  rw [← pow_two, ← pow_two]

--
example (hyp : c = d * a + b) (hyp' : b = a * d) : c = 2 * a * d := by
  rw [hyp, hyp'] -- c = d * a + a * d
  rw [mul_comm] -- c = a * d + a * d
  rw [← two_mul] -- c = 2 * (a * d)
  rw [← mul_assoc] -- c = 2 * a * d


example (a b c : ℕ) (h : a + b = c) : (a + b) * (a + b) = a * c + b * c := by
  nth_rw 2 [h] -- (a+b)*c
  rw [add_mul]
