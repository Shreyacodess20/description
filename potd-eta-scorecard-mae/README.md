# THE ETA SCORECARD

Beginner | metrics-and-evaluation

**Difficulty:** Easy
**Tags:** Metrics & Evaluation

---

### Story

Uber's ETA team tracks one north-star error number every single day: simple, but it has to be
exactly right, since it's the number that goes in front of leadership.

---

### The Math

```
MAE = (1/n) * sum_i |y_i - yhat_i|
```

### Input Format

```
n
y_1 yhat_1
...
y_n yhat_n
```

### Output Format

Scalar MAE, 6 decimals.

### Constraints

- `1 <= n <= 10^6`
- Time limit: 1.0 second.

---

### Example

**Input**

```
4
10 12
15 14
20 18
12 13
```

**Output**

```
1.500000
```
