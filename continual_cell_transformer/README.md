# Continual Cell Transformer V4

A research prototype for continual learning with **one shared recurrent cell population** inside a causal RoPE Transformer.

The banked architecture has been removed. There are no addition, multiplication, chemistry, or other manually named task blocks. Concepts must emerge as overlapping activation patterns in the same population.

## Architecture

```text
Embedding
  ↓
Causal attention with RoPE
  ↓
Shared threshold-routed recurrent cells
  ↓
Main FFN
```

Each cell owns a key, activation threshold, read vector, write vector, bias, and sparse recurrent incoming links.

- Cells activate independently; there is no global top-k competition.
- Adding a cell cannot push an old cell out of the route.
- New cells receive old-to-new links, but allocation creates no new-to-old links.
- New cells start with exactly zero write vectors, so allocation has zero immediate effect on existing outputs.
- Consolidated cells can be frozen while recruited cells remain plastic.
- Retention replay and an old-task output penalty discourage new cells from interfering with earlier concepts.

## Checkpoint boundary

V1-V3 banked checkpoints are intentionally unsupported. Retrain the base task once using V4.

## Base addition training

```bash
python train.py \
  --train-file data/addition_train.txt \
  --eval-file data/addition_eval.txt \
  --out-dir runs/addition_v4 \
  --steps 2000 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --eval-batches 20 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.001 \
  --d-model 128 \
  --layers 4 \
  --heads 4 \
  --d-ff 512 \
  --max-cells 256 \
  --initial-cells 64 \
  --embedding-lr 2e-4 \
  --attention-lr 2e-4 \
  --ffn-lr 2e-4 \
  --cell-lr 5e-4 \
  --other-lr 2e-4 \
  --weight-decay 0 \
  --consolidate-active-cells
```

## Continual multiplication update

```bash
python train.py \
  --resume runs/addition_v4/checkpoint.pt \
  --train-file data/multiplication_train.txt \
  --eval-file data/multiplication_eval.txt \
  --retention-file data/addition_eval.txt \
  --out-dir runs/multiplication_v4 \
  --allocate-cells 16 \
  --allocation-seed-batches 8 \
  --steps 1200 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --eval-batches 20 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.001 \
  --embedding-lr 0 \
  --attention-lr 0 \
  --ffn-lr 0 \
  --other-lr 0 \
  --cell-lr 5e-4 \
  --consolidated-cell-scale 0 \
  --retention-replay-weight 1.0 \
  --plastic-sparsity-weight 0.02 \
  --retention-output-penalty 10.0 \
  --weight-decay 0 \
  --consolidate-active-cells
```

The trainer verifies the zero-impact allocation invariant before training and aborts if new cells alter the old logits by more than `1e-5`.

## Chat and inspect activity

```bash
python chat.py \
  --checkpoint runs/multiplication_v4/checkpoint.pt \
  --show-routing
```

Telemetry includes population coverage, plastic-cell activity, plastic output RMS, active/consolidated/plastic counts, and the most active cell IDs.

## What counts as success

- multiplication held-out loss falls;
- addition retention loss stays close to its pre-update value;
- newly recruited cells contribute on multiplication;
- their output remains near zero on addition;
- exact-answer accuracy remains high on both tasks.

This remains an experimental architecture, not evidence of human-like learning.
