# Dot product and vector norms (L1, L2, L-infinity)

Beginner | linear-algebra | vectors

### The problem, from first principles

Say two friends rank the same five restaurants, one through five. How similar are their tastes? One way to answer: multiply their scores for each restaurant together and add the products up. If they agree (both love the same places, both hate the same places), the products are all large and positive; if they disagree, some products come out negative and cancel the rest. That single number, "multiply corresponding entries, then add," is the dot product, and it is doing real work here: it's a similarity score, computed without ever explicitly comparing "restaurant 1 vs restaurant 1."

Separately: how do you measure how "big" a vector is at all? "Big" turns out not to have one answer. A weight vector with one huge entry and the rest zero, and one with many small entries that sum to the same total, can be equally "big" by one measure and wildly different by another. Three specific ways of measuring size, L1, L2, L-infinity, show up constantly once you start regularizing models and clipping gradients, and this question is where you build the vocabulary for all three.

### From theory to code

Theory gives the exact formula for the dot product (sum of elementwise products) and for each norm. Implement all four as simple, direct translations of those formulas, no loops.

Implement `dot_product`, `l1_norm`, `l2_norm` and `linf_norm` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `a` and `b` (for `dot_product`) are 1D arrays of the same length.
- Every function returns a plain Python `float`, not a 0-d NumPy array.
- No explicit Python loop over vector entries anywhere.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`dot_product` is exactly `elementwise_multiply` (an earlier question) followed by a sum over everything.

</details>

<details>
<summary>Hint 2</summary>

`l2_norm(x)` and `dot_product(x, x)` are related by a single square root, you don't have to derive it from scratch twice.

</details>
