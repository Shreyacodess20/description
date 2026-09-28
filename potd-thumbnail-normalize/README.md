# THUMBNAIL NORMALIZE

Beginner | computer-vision | data-processing

**Difficulty:** Easy
**Tags:** Computer Vision, Data Processing

---

### Story

Netflix normalizes every uploaded artwork candidate before it reaches the thumbnail-ranking CNN.
Get the normalization wrong and the model silently sees out-of-distribution inputs with no error
thrown anywhere.

---

### The Math

```
x' = (x - mu) / sigma
```

where `mu` and `sigma` are the mean and **population** standard deviation of the pixel values in
the given image.

### Input Format

```
H W
p_1,1 ... p_1,W
...
p_H,1 ... p_H,W
```

### Output Format

First line: `mu` and `sigma`. Then the `H x W` normalized matrix, row by row, all to 6 decimals.

### Constraints

- `1 <= H, W <= 512`
- Time limit: 1.0 second.

---

### Example

**Input**

```
2 2
100 150
200 250
```

**Output**

```
175.000000 55.901699
-1.341641 -0.447214
0.447214 1.341641
```
