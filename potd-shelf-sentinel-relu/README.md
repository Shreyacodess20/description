# SHELF SENTINEL

Beginner | activation-functions | autograd

**Difficulty:** Easy
**Tags:** Activation Functions, Autograd

---

### Story

Walmart's smart-shelf sensors report a raw weight-delta signal every time an item is picked up or
placed back, and that signal is noisy in both directions: small negative jitters from shelf
vibration, and clean positive spikes from genuine removals. Before this feeds the restock-trigger
model, every value below zero must be clamped away. You are implementing the exact ReLU layer used
in the sensor-ingestion pipeline, forward and backward.

---

### The Math

```
f(x) = max(0, x)         forward
f'(x) = 1 if x > 0 else 0   backward
```

Given a batch of raw readings `x` and, separately, an upstream gradient `g` from the next layer,
compute the forward activation and the backward gradient `g * f'(x)`.

---

### Input Format

```
n
x_1 x_2 ... x_n
g_1 g_2 ... g_n
```

### Output Format

Two lines: the activated values, then the backward gradient, each space-separated to 6 decimal
places.

### Constraints

- `1 <= n <= 10^5`
- Time limit: 1.0 second. Memory: 64 MB.

---

### Example

**Input**

```
5
-2.0 0.0 3.0 -0.5 1.5
1.0 1.0 1.0 1.0 1.0
```

**Output**

```
0.000000 0.000000 3.000000 0.000000 1.500000
0.000000 0.000000 1.000000 0.000000 1.000000
```
