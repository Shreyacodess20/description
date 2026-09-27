# Matrix inverse, and when it does not exist

Intermediate | linear-algebra | matrices

### The problem, from first principles

Solving `3x = 6` for `x` is dividing both sides by 3, multiplying by the reciprocal `1/3`. The matrix version of "solve `Ax = b` for `x`" wants the same move: multiply both sides by something that undoes `A`. That something is `A`'s inverse, `A^-1`, and exactly like `1/0` doesn't exist, some matrices have no inverse at all, they've thrown away information a reciprocal-style operation would need to recover.

The question worth answering before ever computing an inverse: how do you know, cheaply, whether one exists? That's what the determinant is for, a single number that tells you, without doing the full inversion, whether `A` can be undone at all.

### From theory to code

Theory ties invertibility to a single condition: `det(A) != 0`. Implement the determinant check first, use it to guard against ever calling `np.linalg.inv` on a matrix that can't be inverted, and only then compute the actual inverse.

Implement `determinant`, `is_invertible` and `matrix_inverse` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `a` is always square.
- `is_invertible` must use a numerical tolerance, not exact equality against `0.0`.
- `matrix_inverse` must raise `np.linalg.LinAlgError` on a singular matrix rather than returning whatever `np.linalg.inv` happens to produce.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.linalg` already has a function that computes the determinant directly, you're wrapping it, not deriving cofactor expansion by hand.

</details>

<details>
<summary>Hint 2</summary>

Floating-point arithmetic almost never produces a bit-exact `0.0` for a genuinely singular matrix's determinant. `np.isclose` is the right tool, `== 0.0` is not.

</details>
