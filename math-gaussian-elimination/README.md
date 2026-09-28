# Solving Linear Systems by Hand: Gaussian Elimination

Intermediate | linear-algebra

### The problem, from first principles

`05-matrix-inverse` computes `A⁻¹` and uses it to solve `Ax = b`, but never explains _how_ an inverse (or a solution) is actually found by hand in the first place — it's presented as a formula to apply, not a procedure. Gaussian elimination is that procedure: a systematic way to reduce a system of equations, step by step, until the answer is directly readable off the result. It's also the exact mechanism behind `11-lu-decomposition` and behind why `05-matrix-inverse`'s inverse fails to exist for some matrices.

### From theory to code

Implement `gaussian_eliminate(A, b)`, reducing the augmented system `[A | b]` to upper-triangular form via row operations, then `back_substitute(U, c)`, solving the resulting triangular system for `x`. The signatures and docstrings are already in the editor.

### Constraints

- `A` is a square `n × n` NumPy array; `b` is a length-`n` NumPy array.
- Assume no row swaps are needed (every pivot encountered is non-zero) — the same assumption `11-lu-decomposition` makes.
- `gaussian_eliminate` returns the reduced augmented system as `(U, c)`: `U` upper-triangular, `c` the correspondingly transformed right-hand side.
- `back_substitute` returns the solution vector `x` such that `Ux = c`.

### Hints

<details>
<summary>Hint 1</summary>

Work on a combined copy of `A` and `b` together (or keep them as two separate arrays updated in lockstep) — every row operation applied to eliminate a variable in `A` must be applied identically to the corresponding entry of `b`, since they represent the same equation.

</details>

<details>
<summary>Hint 2</summary>

For back substitution, solve for the **last** variable first (its row has only one unknown left), then substitute that known value into the row above to solve for the next variable, working upward — the reverse order of how elimination worked downward.

</details>
