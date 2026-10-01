# THE CAPTION SPLITTER

Beginner | nlp

**Difficulty:** Easy
**Tags:** NLP

---

### Story

Snap's caption-moderation pipeline needs a first-pass tokenizer before any heavier model runs.
Every unrecognized word must fall back to a single `<unk>` token, never crash the pipeline.

---

### The Math

Given a fixed vocabulary mapping word to id (id `0` reserved for `<pad>`, id `1` reserved for
`<unk>`), split the input text on whitespace and map each word to its id, using `<unk>`'s id for any
out-of-vocabulary word.

### Input Format

```
V
word_1 id_1
...
word_V id_V
caption text on one line
```

### Output Format

Space-separated token ids, one line.

### Constraints

- `1 <= V <= 10^4`, caption up to 500 characters
- Time limit: 1.0 second.

---

### Example

**Input**

```
3
the 2
cat 3
sat 4
the cat sat on mat
```

**Output**

```
2 3 4 1 1
```
