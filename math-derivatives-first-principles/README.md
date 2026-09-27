# Derivatives from first principles: the limit definition, computed numerically

Beginner | calculus

### The problem, from first principles

You're driving and want to know your speed at this exact instant, not your average speed over the whole trip. Speed is "distance per unit time", so a natural first attempt: look at the distance covered over a very short time window, and divide. Make that window shorter and shorter, and the answer stabilizes on your true instantaneous speed. That stabilizing process, "measure over an ever-shrinking window", is precisely what a derivative is, and precisely what a computer can approximate without ever taking an actual limit.

This isn't just a warm-up exercise: it's the tool every "did I compute this gradient correctly" check in this curriculum relies on. Every backward pass written from here on (`03-tanh`'s, `linear_regression`'s gradient, the entire Autograd track) can be sanity-checked by comparing its analytical answer against exactly the numerical approximation you build here.

### From theory to code

Theory gives two ways to approximate the limit with a small, finite step: one stepping only forward from `x`, one stepping symmetrically in both directions. Implement both, directly translating their formulas.

Implement `forward_difference(f, x, eps=1e-5)` and `central_difference(f, x, eps=1e-5)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `f` is a plain Python function taking and returning a single number.
- `eps` defaults to `1e-5`, don't hardcode a different value.
- No symbolic differentiation, no calling a derivative library, only function evaluations of `f`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Both functions only ever call `f` at most twice. If you're calling it more than that, you've overcomplicated it.

</details>

<details>
<summary>Hint 2</summary>

`central_difference` steps by `eps` in both directions and averages the "rise" over `2 * eps`, not `eps`, since the total distance covered is twice the single step.

</details>
