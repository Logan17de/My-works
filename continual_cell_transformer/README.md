# Continual Cell Transformer V4

A research prototype for continual learning with one shared recurrent cell
population inside a causal Transformer.

V4 removes named routing banks. Addition, multiplication, chemistry and other
concepts are not assigned separate blocks. They must emerge as overlapping
activation patterns inside the same population.

## Core mechanism

After attention:

1. Every active cell compares its key with the contextual token state.
2. A cell activates only when its own learned threshold is exceeded.
3. Active cells communicate through sparse recurrent links.
4. Their latent outputs are added back to the Transformer state.
5. New concepts may recruit dormant cells.

There is no global top-k competition. Adding a cell cannot push an old cell out
of the route.

New cells start with zero write vectors, so allocation has no immediate effect
on the model output. They receive inbound links from established cells but no
new-to-old links.

## Files

- `config.py` — Transformer and shared-population settings
- `model.py` — RoPE Transformer and threshold-routed recurrent cells
- `tokenizer.py` — UTF-8 fallback plus append-only token insertion
- `train.py` — base and continual training, replay and retention metrics
- `chat.py` — deterministic chat with shared-population telemetry

## Important checkpoint boundary

V4 changes routing from top-k/banks to independent thresholds. Old V1–V3
checkpoints can be loaded for inspection, but their exact old routing is not
preserved. Controlled experiments should retrain the base task with V4.

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
  --recurrent-fan-in 8 \
  --cell-maturity-steps 2000 \
  --embedding-lr 2e-4 \
  --attention-lr 2e-4 \
  --ffn-lr 2e-4 \
  --cell-lr 5e-4 \
  --other-lr 2e-4 \
  --mature-cell-scale 1.0 \
  --weight-decay 0 \
  --seal-active-cells
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
  --steps 1000 \
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
  --cell-lr 1e-3 \
  --mature-cell-scale 0 \
  --retention-replay-weight 1.0 \
  --retention-output-penalty 10.0 \
  --activity-sparsity-weight 0.01 \
  --weight-decay 0 \
  --seal-active-cells
```

## Chat and inspect activity

```bash
python chat.py \
  --checkpoint runs/multiplication_v4/checkpoint.pt \
  --show-routing
```

The routing output now reports active shared-population cell IDs, activity
fraction, mean gate strength and the contribution from newly recruited cells.

## What counts as success

- multiplication held-out loss falls;
- addition retention loss stays near its pre-update value;
- recruited cells contribute strongly on multiplication;
- their output remains near zero on addition;
- addition and multiplication exact-answer accuracy both remain high.

This remains an experimental architecture, not evidence of human-like learning.
