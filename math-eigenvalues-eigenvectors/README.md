# Eigenvalues and eigenvectors of a small matrix

Advanced | linear-algebra | matrices

### The problem, from first principles

Apply a matrix to a vector and, in general, you get a vector pointing in some new direction, rotated and rescaled both. But every matrix has special directions where nothing rotates: feed it a vector already pointing that way, and you get back the same direction, only longer or shorter. Those special, rotation-immune directions are eigenvectors, and how much they stretch is the eigenvalue.

Why bother finding them? Because they tell you what a matrix _actually does_ to space, stripped of the confusing mix of rotation and scaling every other direction shows. A covariance matrix's eigenvectors, for instance, point along the directions your data varies the most and the least, exactly what PCA (a later question) exploits to compress data without losing much information.

### From theory to code

Theory restricts to real symmetric matrices specifically, where eigenvalues are guaranteed real and a specialized, numerically stable routine exists. Implement the decomposition using that routine, and a separate check that verifies the defining equation directly for any claimed eigenvalue/eigenvector pair.

Implement `eigen_decomposition(a)` and `verify_eigenpair(a, eigenvalue, eigenvector)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `a` is always real and symmetric (`a == a.T`).
- Use the specialized symmetric-matrix routine, not the general-purpose one, Theory explains why they differ.
- `verify_eigenpair` checks the actual defining equation, not just that `v` "looks like" an eigenvector.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

NumPy has two different eigen-decomposition functions, one general, one specifically for symmetric/Hermitian matrices. The specialized one is faster, more stable, and always returns real values, exactly what this question wants.

</details>

<details>
<summary>Hint 2</summary>

The defining equation is `A @ v = lambda * v`. Compute both sides and compare them, that's the whole check.

</details>
