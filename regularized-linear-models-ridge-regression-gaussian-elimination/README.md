# Ridge Regression From Scratch: Solving the Normal Equation by Hand

Advanced | classical-ml | regression | numerical-methods

### The problem, from first principles

`Ridge Regression (L2)` already derives the closed-form solution `theta = (X^T X + alpha I)^-1 X^T y` and hands the actual inversion to `np.linalg.inv`. That's the right call for a normal library function, but it skips the part that makes the formula reliable in the first place: `np.linalg.inv` is itself just Gaussian elimination under the hood, and elimination done naively (always using whichever row is "next," never checking its size) can divide by a pivot that's zero, or merely tiny, and either crash outright or quietly produce garbage from floating-point cancellation.

This question asks for the whole pipeline by hand: assemble ridge's normal equation yourself, then solve it with your own Gaussian elimination, one that picks the LARGEST available pivot at each step (partial pivoting) instead of whatever row happens to be next. No `np.linalg.inv`, `np.linalg.solve`, or `np.linalg.pinv` anywhere in your solution, the whole point is building the thing those functions hide.

### From theory to code

Implement two functions. `gaussian_elimination_solve(a, b)` solves the square linear system `a @ x = b` via elimination with partial pivoting, no numpy solver shortcuts. `ridge_regression_predict(input, target, lam, queries)` builds ridge's normal equation and calls your own solver to fit it, then predicts for every row of `queries`.

One convention this question uses that `Ridge Regression (L2)` does NOT: the bias column here is regularized too, no zeroing out the last diagonal entry. Simpler to reason about, and it matches how several real systems (including plain L2 weight decay applied uniformly) actually do it, contrasting with the convention seen there is itself part of the point.

### Constraints

- Append the bias column as the LAST column of the augmented design matrix (a column of ones), matching `Linear Regression: closed form (Normal Equation)`'s own convention.
- `lam` regularizes every dimension of the normal equation's identity term, including the bias position, unlike `Ridge Regression (L2)`.
- `gaussian_elimination_solve` must pivot: at each elimination step, swap in the row with the largest-magnitude entry in the current column before eliminating. A matrix that's perfectly solvable but has a zero (or tiny) pivot in its natural row order must still work.
- `lam=0` degenerates to plain least squares, solved through the same elimination path (no special-casing).

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`gaussian_elimination_solve`: work on the augmented matrix `[a | b]`. At column `k`, find the row (at or below `k`) with the largest absolute value in that column, swap it into row `k`, then eliminate that column from every row below as usual. Back-substitute once the matrix is upper-triangular.

</details>

<details>
<summary>Hint 2</summary>

`ridge_regression_predict`: augment `input` with a ones column (`np.hstack`), form `a = X^T X + lam * I` and `b = X^T y` (full-size identity, no zeroing), call `gaussian_elimination_solve(a, b)` to get the combined weight+bias vector, then augment `queries` the same way and take the dot product with the solved vector.

</details>
