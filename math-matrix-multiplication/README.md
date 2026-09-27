# Matrix multiplication from first principles

Intermediate | linear-algebra | matrices

### The problem, from first principles

`01-vectors-matrices-tensors` already warned that `elementwise_multiply` is not matrix multiplication. Here's why they're genuinely different operations, not just different names for the same thing.

Say a small store sells 3 products, and you track how many units of each product 4 different customers bought, a `(4, 3)` matrix. Separately, each product has a price, a length-3 vector. "How much did each customer spend in total" is not an elementwise operation, customer 1's spending depends on ALL three of their quantities combined with ALL three prices, not one quantity paired with one price. That "combine an entire row with an entire column" operation, repeated for every customer, is matrix multiplication, and every entry of the result is its own independent dot product.

### From theory to code

Theory derives the shape rule (`a`'s columns must equal `b`'s rows) and the exact formula for one output entry: row `i` of `a`, dotted against column `j` of `b`. Implement that directly, one dot product per output position, using explicit loops over positions rather than `np.matmul`/`@` so the row/column mechanism is visible in the code itself, once, before you rely on a library to do it silently for the rest of the curriculum.

Implement `matmul_from_scratch(a, b)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `a` is `(m, k)`, `b` is `(k, n)`, result is `(m, n)`.
- Raise `ValueError` if `a`'s columns don't match `b`'s rows.
- No `np.matmul`, `np.dot` on the full matrices, or `@` anywhere in your implementation, loop over output positions `(i, j)` explicitly.
- The inner reduction (summing one row-times-column product) may use NumPy's own elementwise multiply and sum, reimplementing that with a third loop isn't the point of this question.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Every entry `result[i, j]` only ever needs `a`'s row `i` and `b`'s column `j`, nothing else from either matrix.

</details>

<details>
<summary>Hint 2</summary>

`a[i, :] * b[:, j]` lines up two same-length 1D vectors elementwise. What single reduction turns that into the number that goes in `result[i, j]`?

</details>
