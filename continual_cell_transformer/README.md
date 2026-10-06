# Continual Cell Transformer V7

**Project status: paused.** The implementation, tests, dataset generators, and measured results are preserved on this branch. See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for the complete research record and stopping point.

## Goal

This prototype tests continual learning without explicit task IDs or manually assigned task banks. It uses one shared Transformer and one shared population of dynamically routed cells that can expand internal and outer capacity over time.

## Architecture

V7 combines:

1. **Weight-tied recurrent Transformer depth** — one causal Transformer block repeats from `min_depth` to `max_depth`.
2. **Adaptive inference halting** — a learned halt head can stop inference early after cumulative halt mass reaches the configured threshold.
3. **Independent cell routing** — each cell has its own key and threshold; there is no global top-k competition.
4. **Growable cell interiors** — each cell starts with a few active micro-neurons and can recruit more.
5. **Dormant outer capacity** — unused cells can be activated when existing capacity is insufficient.
6. **Consolidation and replay** — mature capacity can receive reduced gradients while old-task examples are replayed.
7. **Zero-impact growth checks** — structural insertion is rejected when it immediately changes logits beyond tolerance.

Each recurrent pass is:

```text
hidden state
  -> causal attention
  -> shared routed cell population
  -> FFN
  -> halting decision
  -> next recurrent depth or output
```

## Important entry points

| File | Purpose |
|---|---|
| `model.py` | Recurrent Transformer and adaptive halting |
| `cells.py` | Shared routed cell population and micro-neurons |
| `train_fixed.py` | Fresh arithmetic training with `math_answer_eos_v2` |
| `train_continual_math.py` | Continual arithmetic update with retention and growth |
| `evaluate_math_checkpoint.py` | Exact arithmetic checkpoint evaluation |
| `generate_math_generalization_datasets.py` | Held-out operand and formatting benchmark |
| `evaluate_math_generalization.py` | Generalization report with sampled quick mode |
| `generate_column_addition_datasets.py` | Procedural two-digit addition benchmark |
| `train_column_addition.py` | Procedural column-addition trainer |
| `evaluate_column_addition.py` | Final-answer, procedure, and sub-step evaluation |
| `debug_zero_impact_growth.py` | Structural growth regression test |

## Measured result

The strict held-out-operand addition benchmark produced:

| Split | Accuracy |
|---|---:|
| Seen canonical facts | 100.00% |
| Familiar facts with unseen whitespace | 98.46% |
| Any fact containing held-out operand `7` | 0.00% |
| Held-out operand plus unseen whitespace | 0.00% |
| Parentheses/leading-zero notation | 49.38% |
| Two-digit operands | 0.00% |

This establishes that the current system learned a format-tolerant lookup mechanism rather than a reusable arithmetic rule.

Other observed behaviors:

- original single-digit addition reached `100/100` exact mastery on trained facts,
- inference depth became variable rather than always using the maximum depth,
- continual multiplication recruited 2 outer cells and 29 micro-neurons,
- multiplication remained incomplete because some answers repeated instead of terminating,
- the prepared procedural column-addition benchmark was not trained before the project was paused.

## Reproduce the generalization benchmark

Generate data:

```bash
python generate_math_generalization_datasets.py \
  --heldout-digit 7 \
  --repeats 50 \
  --seed 17
```

Train a fresh model:

```bash
python train_fixed.py \
  --train-file data/addition_generalization_train.txt \
  --eval-file data/addition_seen_canonical_eval.txt \
  --out-dir runs/addition_generalization_v7 \
  --steps 3000 \
  --batch-size 32 \
  --seq-len 32 \
  --eval-interval 50 \
  --mastery-stop \
  --mastery-accuracy 1.0 \
  --mastery-patience 2 \
  --d-model 128 \
  --heads 4 \
  --d-ff 512 \
  --min-depth 2 \
  --max-depth 8 \
  --max-cells 256 \
  --initial-cells 64 \
  --max-micro-neurons 16 \
  --initial-micro-neurons 4 \
  --embedding-lr 2e-4 \
  --attention-lr 2e-4 \
  --ffn-lr 2e-4 \
  --cell-lr 5e-4 \
  --halt-lr 1e-4 \
  --other-lr 2e-4 \
  --depth-penalty 0.005 \
  --routing-loss-weight 0.05 \
  --weight-decay 0 \
  --grad-clip 1.0 \
  --consolidate-active-cells
```

Run a fast sampled report:

```bash
python evaluate_math_generalization.py \
  --checkpoint runs/addition_generalization_v7/checkpoint.pt \
  --operation addition \
  --heldout-digit 7 \
  --max-new-tokens 3 \
  --limit-per-split 25 \
  --seed 17
```

Remove `--limit-per-split` for the complete report.

## Research boundary

This repository is a research prototype. Current evidence supports claims about memorization, sparse routing, variable-depth inference, and mechanical structural growth. It does not support claims of arithmetic reasoning, human-like continual learning, or general algorithm acquisition.
