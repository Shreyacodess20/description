# SENSOR FUSION LAYER

Beginner | neural-networks

**Difficulty:** Easy
**Tags:** Neural Networks

---

### Story

A single linear layer fuses concatenated radar and camera features before Tesla's perception stack
does anything more sophisticated: the "hello world" layer of the whole pipeline, but it has to be
bit-exact.

---

### The Math

```
y = W x + b
```

### Input Format

```
d_in d_out
W (d_out x d_in, row-major)
b (d_out values)
x (d_in values)
```

### Output Format

`y`, `d_out` values, 6 decimals.

### Constraints

- `1 <= d_in, d_out <= 256`
- Time limit: 1.0 second.

---

### Example

**Input**

```
3 2
1 0 1
0 1 1
0.5 -0.5
1 2 3
```

**Output**

```
4.500000 4.500000
```
