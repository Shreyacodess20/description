# Transpose, and its role in reshaping without copying data

Beginner | linear-algebra | matrices

### The problem, from first principles

`03-matrix-multiplication` required `a`'s columns to match `b`'s rows. Real data doesn't always arrive in the shape an operation needs: `weight` in a linear layer is stored as `(out_features, in_features)`, one row per output neuron, but matrix-multiplying it against a `(batch, in_features)` input needs `in_features` to line up as the _shared_ dimension on both sides. Something has to flip `weight`'s two axes first, without which `linear` (this curriculum's very first question) simply couldn't be written as one matmul.

That flip is the transpose. The part that makes it worth its own question, beyond "flip the axes", is that a correct implementation does this essentially for free: no new memory allocated, no numbers moved, just a different way of reading the same bytes.

### From theory to code

Theory defines the transpose (`x[i, j] -> x.T[j, i]`) and explains, at the memory level, why it costs nothing. Implement `transpose`, and a helper that verifies the "no copy" claim directly by checking shared memory.

Implement `transpose(x)` and `is_a_view_of(original, derived)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `transpose` only needs to handle 2D input (a full N-D `.transpose(dims)` is a later Deep Learning Core question).
- `is_a_view_of` must check actual shared memory, not just equal values, two separately-allocated arrays holding identical numbers are not "the same view."

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

NumPy arrays already expose the transpose as a one-character attribute. You are not asked to build it from index arithmetic.

</details>

<details>
<summary>Hint 2</summary>

There's a NumPy function whose entire job is answering "do these two arrays' buffers overlap," which is exactly what "is this a view" means.

</details>
