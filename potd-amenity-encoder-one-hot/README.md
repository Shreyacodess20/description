# AMENITY ENCODER

Beginner | data-processing | classical-ml

**Difficulty:** Easy
**Tags:** Data Processing, Classic ML

---

### Story

Every Airbnb listing carries a categorical amenities tag, and the pricing model cannot consume raw
strings. This is the encoding step that runs before every single downstream feature touches the
model.

---

### The Math

One-hot encode a categorical column of `n` values into an `n x k` binary matrix, where `k` is the
number of **distinct** categories present, columns ordered alphabetically.

### Input Format

```
n
cat_1
cat_2
...
cat_n
```

### Output Format

First line: the `k` distinct categories, alphabetically, space-separated. Next `n` lines: the
one-hot row for each input, space-separated `0`/`1`.

### Constraints

- `1 <= n <= 10^4`, category strings up to 20 characters
- Time limit: 1.0 second.

---

### Example

**Input**

```
5
wifi
parking
wifi
pool
parking
```

**Output**

```
parking pool wifi
0 0 1
1 0 0
0 0 1
0 1 0
1 0 0
```
