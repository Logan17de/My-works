# Continual Cell Transformer V7 — Experiment Status

**Status:** Paused on 2026-08-06. The implementation and measured findings are preserved for future work.

This document records what was actually implemented, what was measured, what remains unverified, and why the current experiment was stopped. Generated datasets, run directories, model checkpoints, and local logs are intentionally not committed.

## Research question

Can a Transformer continually acquire new behavior through a shared, dynamically routed and structurally expandable cell population, without explicit task IDs or manually assigned task banks?

A second question emerged during testing: does success on arithmetic examples reflect a reusable arithmetic procedure, or only storage of input-answer mappings?

## V7 architecture

V7 uses one causal Transformer block repeatedly from `min_depth` through `max_depth`.

Each recurrent pass performs:

1. causal self-attention,
2. a shared routed cell population,
3. a feed-forward network,
4. a learned halting decision.

The cell population has:

- independent threshold routing for each cell,
- no global top-k selection,
- no named task/domain banks,
- overlapping sparse activation patterns,
- recurrent communication between cells,
- expandable micro-neurons inside each cell,
- dormant outer cells that can be recruited,
- consolidation masks and reduced gradients for mature capacity.

The model supports zero-impact structural growth: newly activated write vectors start at zero, and a deterministic before/after logit check aborts growth when drift exceeds the configured tolerance.

## Important implementation fixes

### Arithmetic objective

The original arithmetic objective also rewarded template and control-token prediction. The corrected `math_answer_eos_v2` objective supervises:

- answer digits with weight `1.0`,
- EOS with weight `0.01`,
- prompt, newline, and `<END>` tokens with weight `0.0`.

The low EOS weight prevented formatting tokens from dominating the loss, but it also made termination weaker than answer-digit learning in some multiplication examples.

### Exact mastery evaluation

Mastery uses greedy generation and exact string equality between the extracted integer and expected answer. It retains raw completions and per-generated-token depth traces for representative errors.

### Zero-impact micro-neuron growth

Two structural-growth bugs were fixed:

1. Dividing by `sqrt(active_micro_count)` changed existing outputs whenever a zero-output micro-neuron was activated. V7 uses the fixed initial micro-neuron count as the normalization denominator.
2. Advanced-index calls such as `tensor[index_list].zero_()` can modify a copy rather than the underlying parameter. Growth initialization now uses explicit assignment for micro-neuron and recurrent-edge rows.

### Evaluation runtime

The generalization evaluator performs autoregressive generation separately for each example. Progress messages and `--limit-per-split` were added because a complete run can require thousands of sequential recurrent forwards.

## Measured experiments

### 1. Addition mastery on the original table

The model was trained on all ordered single-digit addition pairs from `0 + 0` through `9 + 9`.

Observed result:

- exact mastery reached `100/100` on the same fact set,
- inference depth became variable rather than always using depth 8,
- cell routing remained sparse at roughly 10–11 active cells from a 64-cell population.

This result demonstrates storage and retrieval of the trained mappings. It does not demonstrate arithmetic-rule generalization because evaluation reused the trained facts.

### 2. Continual multiplication after addition

A multiplication update resumed from the mastered addition checkpoint with:

- very low backbone learning rates,
- normal learning rate for plastic cell capacity,
- retention replay on addition,
- reduced gradients for consolidated cells,
- autonomous micro-neuron and outer-cell growth.

Observed pool after the run:

```text
active cells:          66
consolidated cells:    66
reserve cells:         190
active micro-neurons:  285
consolidated micro:    285
plastic micro:         0
```

Relative to the initial 64 cells and 256 active micro-neurons, the run recruited 2 outer cells and 29 micro-neurons.

Manual examples included correct retained addition and learned multiplication facts:

```text
0 * 2 = 0
1 * 2 = 2
6 * 5 = 30
9 + 9 = 18
0 + 2 = 2
```

However, multiplication mastery was not complete. For example:

```text
9 * 9 = 81818181818
```

The answer digits `81` appeared, but generation failed to terminate and repeated the pattern. Therefore this run is recorded as partial, not as mastered multiplication.

### 3. Adaptive depth behavior

Training calls the model with adaptive early exit disabled, so every training example executes all configured recurrent depths. The reported training `expected_depth` is the expectation of the soft halting distribution, not the number of recurrent passes executed.

During inference, hard early exit became variable. Examples observed after continual training included:

```text
0 * 2: depth 3
1 * 2: depth 3
9 + 9: depth 4
6 * 5: depth 5
9 * 9: depth 6
```

This demonstrates functional variable-depth inference. It does not establish that depth tracks difficulty or correctness.

### 4. Strict arithmetic generalization test

A fresh addition model was trained with operand digit `7` completely excluded from training. All other ordered single-digit pairs appeared in four spacing formats. Evaluation separated familiar facts, unseen formatting, held-out operands, notation changes, and two-digit extrapolation.

Complete measured report:

| Split | Correct | Accuracy | Average depth |
|---|---:|---:|---:|
| Seen canonical | 81/81 | 100.00% | 3.835 |
| Seen, unseen whitespace | 319/324 | 98.46% | 3.850 |
| Held-out operand 7 | 0/19 | 0.00% | 3.262 |
| Held-out operand 7 + unseen whitespace | 0/76 | 0.00% | 3.382 |
| Parentheses/leading-zero notation | 80/162 | 49.38% | 3.764 |
| Two-digit operands | 0/100 | 0.00% | 3.976 |

Representative failures:

```text
0 + 7 -> 0        expected 7
1 + 7 -> 1        expected 8
2 + 7 -> 5        expected 9
10 + 10 -> 1      expected 20
10 + 11 -> 1      expected 21
10 + 12 -> 2      expected 22
```

## Main conclusion

The current V7 experiment demonstrates:

- strong memorization of trained arithmetic mappings,
- high robustness to unseen whitespace around familiar facts,
- sparse routed cell activity,
- working variable-depth inference,
- mechanical structural expansion,
- partial continual acquisition with retained manual addition examples.

It does **not** demonstrate:

- a reusable addition rule,
- generalization to a completely unseen operand,
- extrapolation from single-digit to two-digit arithmetic,
- halting based on uncertainty,
- complete multiplication mastery,
- protection from all functional interference through the shared residual stream and backbone.

The strict generalization result is decisive: seen facts remained at 100%, while every fact involving the held-out operand failed. The learned behavior is therefore dominated by a format-tolerant lookup mechanism rather than a general arithmetic procedure.

## Procedural benchmark prepared but not run

A two-digit column-addition benchmark was implemented as the proposed next experiment. It:

- splits unordered operand pairs to prevent commutative train/test leakage,
- supervises a compact ones/carry/tens/final-answer procedure,
- evaluates final-answer, full-procedure, and individual sub-step accuracy,
- separates seen pairs, unseen pairs, carry, no-carry, and unseen-format splits.

Example target:

```text
18 + 27
O8+7=15,D5,C1|T1+2+1=4,D4,C0|H0|A45
```

The project was paused before this benchmark was trained. Its code is included for reproducibility, but there are no measured procedural results.

## Unresolved research issues

1. **Shared residual interference:** new cells modify the same hidden stream consumed by later attention, FFN, routing, and output layers.
2. **Soft-training/hard-inference mismatch:** training optimizes a weighted depth mixture while inference may emit from an individual early-exit state.
3. **Intermediate-depth readiness:** no direct answer loss guarantees that every possible exit depth is independently correct.
4. **Straight-through routing leakage:** hard forward gates use surrogate soft gradients during training.
5. **Growth criteria:** relevance, utilization, gradient pressure, and contribution are heuristic proxies rather than proof that new capacity is required.
6. **Arithmetic inductive bias:** routed cells alone did not produce algorithmic arithmetic from a fact table.
7. **Termination learning:** the low EOS weight can allow correct digits followed by repetitive continuation.

## Recommended restart point

A future continuation should begin with the prepared procedural column-addition benchmark, but first add:

- direct auxiliary answer losses at each recurrent depth,
- separate reporting for adaptive and forced-full-depth accuracy,
- batched generation for faster evaluation,
- stronger but balanced EOS supervision,
- a frozen-backbone ablation to isolate cell-only plasticity,
- a plain Transformer baseline with the same parameter count and data split.

No claim of arithmetic reasoning or human-like continual learning should be made from the current results.
