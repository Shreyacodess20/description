# Vector Projection and Orthogonal Decomposition

Beginner | linear-algebra

### The problem, from first principles

"How much of vector `a` points in the same direction as vector `b`?" comes up constantly: it's the geometric idea behind a shadow, behind "how correlated are these two signals," and it's the exact operation `10-gaussian-elimination` and `12-gram-schmidt` both build on to remove one vector's overlap with another. This question implements it directly, before either of those questions needs it as a building block.

### From theory to code

Implement `project(a, b)`, returning the projection of `a` onto `b` — the component of `a` that points along `b`'s direction — then `orthogonal_component(a, b)`, the leftover part of `a` that's perpendicular to `b`. The signatures and docstrings are already in the editor.

### Constraints

- `a` and `b` are 1D NumPy arrays of the same length.
- `b` is never the zero vector (projecting onto a direction that doesn't exist is undefined).
- `project(a, b) + orthogonal_component(a, b)` must reconstruct `a` exactly (up to floating-point precision).

### Hints

<details>
<summary>Hint 1</summary>

The projection's length along `b` is `(a · b) / (b · b)` — a dot product ratio, not the dot product alone. Multiply that scalar by the vector `b` itself to get the actual projection vector.

</details>

<details>
<summary>Hint 2</summary>

Once you have `project(a, b)`, the orthogonal component is just `a` minus it — whatever's left over after removing the along-`b` part must be everything perpendicular to `b`.

</details>
