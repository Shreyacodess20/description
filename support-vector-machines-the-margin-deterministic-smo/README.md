# THE MARGIN

Advanced | classical-ml | svm | optimization | quadratic-programming | production-ml

**Difficulty:** Hard
**Topic:** Deterministic Sequential Minimal Optimization (SMO), Constrained Quadratic Programming, Production ML

---

### Story

You work on the fraud-detection team at a payments company. You've settled on a linear
max-margin classifier (SVM) as the model — it's simple, fast at inference time, and legally
easy to explain to auditors. But compliance has one hard requirement: the trained model must
be **bit-for-bit reproducible**. Standard SMO implementations (like libsvm) pick which pair
of variables to optimize using heuristics that involve iteration order and tie-breaking that
differ across library versions — unacceptable when a regulator asks you to retrain last
quarter's model and get the _exact same_ decision boundary back.

So your team defines its own fully deterministic training procedure: a **simplified,
cyclic-pairing SMO** — no randomness, no heuristic pair selection, just a fixed rule anyone
can re-implement and get identical results. Your job is to implement it exactly as specified
below, train it on the labeled transaction data, and report the model's raw decision score on
new transactions.

---

### Input Format

```
n d
x_1,1 x_1,2 ... x_1,d y_1
x_2,1 x_2,2 ... x_2,d y_2
...
x_n,1 x_n,2 ... x_n,d y_n
C tol max_passes
m
q_1,1 q_1,2 ... q_1,d
...
q_m,1 q_m,2 ... q_m,d
```

- `y_i ∈ {-1, +1}` (integers)
- `C` — box constraint, `0 < C ≤ 1000`
- `tol` — KKT violation tolerance, `0 < tol ≤ 1`
- `max_passes` — number of consecutive no-change passes required to stop

### Output Format

Print `m` lines: the raw decision score `f(q)` for each query, with at least 6 digits after
the decimal point.

**Judging:** accepted if within `1e-4` absolute or relative error (whichever is larger) of
the reference implementation's output, which follows the algorithm above **exactly**,
including its exact stopping rule — this is intentionally a specific deterministic procedure,
not "run until true convergence to the QP optimum," so implementations that reach the true
optimum by a different path will _not_ generally match.

---

### Constraints

- `1 ≤ n ≤ 200`
- `1 ≤ d ≤ 20`
- `1 ≤ m ≤ 200`
- `1 ≤ max_passes ≤ 1000`
- All feature values satisfy `|value| ≤ 10^4`, given with up to 6 decimal digits.
- No two training feature vectors are identical.
- Time limit: 3 seconds. Memory limit: 256 MB.

---

### Example 1

**Input**

```
4 2
2 2 1
3 3 1
0 0 -1
1 0 -1
1.0 0.001 10
2
2 0
0.5 2
```

**Output**

```
-0.600000
0.400000
```

**Explanation:** Running the deterministic cyclic SMO to convergence (`max_passes = 10`
consecutive clean passes) on this small linearly separable set gives `α = [0.4, 0, 0, 0.4]`
(points `(2,2)` and `(1,0)` become the support vectors) and `b = -1.4`. Then
`f(2,0) = 0.4·1·(2·2+2·0) + 0.4·(-1)·(1·2+0·0) - 1.4 = 0.4·4 - 0.4·2 - 1.4 = -0.6`, and
similarly `f(0.5,2) = 0.4`.

---

### Example 2

**Input**

```
4 2
2 2 1
3 3 1
0 0 -1
1 0 -1
0.25 0.001 5
2
2 0
0.5 2
```

**Output**

```
-0.625000
0.000000
```

**Explanation:** Same data, but now `C = 0.25` is small enough to actively clip both support
vectors: the algorithm converges with `α = [0.25, 0, 0, 0.25]`, `b = -1.125`. This shows the
box constraint binding — with the earlier `C=1.0` the unconstrained-by-box optimum needed
`α = 0.4` per support vector, so lowering `C` below that forces clipping and changes both `b`
and every downstream prediction, even though the _support vectors themselves_ (which training
points end up with nonzero `α`) are unchanged.
