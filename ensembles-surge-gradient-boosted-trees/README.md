# THE SURGE

Advanced | classical-ml | ensembles | gradient-boosting | trees | regularization

**Difficulty:** Medium–Hard
**Topic:** Gradient Boosted Trees (XGBoost-style), Regularized Split Gain, Production ML

---

### Story

You work on the dispatch team at a ride-hailing app. Every few minutes the system needs to
decide how many extra drivers to nudge into each zone before a demand spike hits — arrive
too late and riders wait forever, arrive too early and drivers idle for nothing. Historical
data shows demand is not linear in time-of-day, weather, or nearby event size: it stays flat
for long stretches, then jumps sharply, then plateaus again. That kind of shape is exactly
what gradient boosted regression trees are built for.

You'll implement a minimal but faithful XGBoost-style regressor from scratch — exact greedy
split finding with the regularized gain formula, L2-regularized leaf weights, and
learning-rate shrinkage — and use it to forecast demand at new query points so dispatch can
pre-position drivers before the surge, not after it.

---

### Input Format

```
n d
x_1,1 x_1,2 ... x_1,d y_1
x_2,1 x_2,2 ... x_2,d y_2
...
x_n,1 x_n,2 ... x_n,d y_n
T lambda gamma eta max_depth base_score
m
q_1,1 q_1,2 ... q_1,d
q_2,1 q_2,2 ... q_2,d
...
q_m,1 q_m,2 ... q_m,d
```

- `T` — number of boosting rounds (trees)
- `lambda` — L2 regularization on leaf weights (`≥ 0`)
- `gamma` — minimum gain required to keep a split (`≥ 0`)
- `eta` — learning rate / shrinkage (`0 < eta ≤ 1`)
- `max_depth` — maximum tree depth (root is depth `0`)
- `base_score` — initial prediction for every sample

### Output Format

Print `m` lines: the predicted `ŷ` for each query, with at least 6 digits after the decimal
point.

**Judging:** accepted if within `1e-4` absolute **or** relative error of the reference
solution (whichever is larger). Because split selection is a discrete, exactly-specified
process (fixed tie-breaking rules), the _structure_ of every tree is uniquely determined —
there is no ambiguity to cause divergent predictions if you follow the algorithm exactly.

---

### Constraints

- `1 ≤ n ≤ 500`
- `1 ≤ d ≤ 20`
- `1 ≤ m ≤ 500`
- `1 ≤ T ≤ 30`
- `0 ≤ max_depth ≤ 4`
- `0 ≤ lambda ≤ 10^3`, `0 ≤ gamma ≤ 10^3`, `0 < eta ≤ 1`
- All feature values, targets, and `base_score` satisfy `|value| ≤ 10^4`, given with up to 6 decimal digits.
- Time limit: 2 seconds. Memory limit: 256 MB.

---

### Example 1

**Input**

```
4 1
1 1
2 1
3 10
4 10
1 0 0 1 1 0
2
2
3.5
```

**Output**

```
1.000000
10.000000
```

**Explanation:** `T=1`, `lambda=0`, `gamma=0`, `eta=1`, `max_depth=1`, `base_score=0`. With
`pred_i=0` for all, `g_i = -y_i = [-1,-1,-10,-10]`. Testing the three candidate thresholds
(1.5, 2.5, 3.5), the split at `t=2.5` gives the largest gain (`40.5`), separating
`{1,2}` from `{3,4}`. Leaf weights: left `= -(-2)/2 = 1`, right `= -(-20)/2 = 10`. With
`eta=1` these become the predictions directly. Query `x=2` falls left → `1.0`; query
`x=3.5` falls right → `10.0`.

---

### Example 2

**Input**

```
4 1
1 1
2 1
3 10
4 10
2 1 0 0.5 1 0
2
2
3.5
```

**Output**

```
0.555556
5.555556
```

**Explanation:** Same data, but now `T=2` rounds, `lambda=1`, `eta=0.5`. Round 1 again
splits at `t=2.5` (regularization changes the _magnitude_ of the gain but not which split
wins here), giving leaf weights `2/3` (left) and `20/3` (right); after shrinkage,
`pred` becomes `1/3` for `{1,2}` and `10/3` for `{3,4}`. Round 2 recomputes gradients from
these new predictions, splits at `t=2.5` again, giving leaf weights `4/9` (left) and `40/9`
(right). Final predictions: left `= 1/3 + 0.5·(4/9) = 5/9 ≈ 0.555556`, right
`= 10/3 + 0.5·(40/9) = 50/9 ≈ 5.555556`.
