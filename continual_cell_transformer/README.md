# Continual Cell Transformer V5

V5 combines three adaptive mechanisms in one model:

1. **Adaptive Transformer depth** — one weight-tied Transformer block repeats from `min_depth` to `max_depth`; a learned halting head decides when inference can stop.
2. **Shared expandable cell population** — cells activate independently through learned thresholds. There are no named task banks and no global top-k competition.
3. **Expandable cell interiors** — each cell starts with a small number of active micro-neurons and can activate more internal capacity later.

## Learning policy

Prediction loss trains the model. A depth penalty rewards the cheapest sufficient computation. Loss plateaus plus high population coverage can recruit new cells. Loss plateaus plus saturated internal capacity can activate new micro-neurons. Both forms of growth start with zero output vectors, so allocation is checked to have no immediate effect on existing logits.

The Transformer backbone remains slightly plastic through separate low learning rates, while new cells and micro-neurons learn much faster. Consolidated cell rows receive a configurable gradient scale.

## Base training

```bash
python train.py \
  --train-file data/addition_train.txt \
  --eval-file data/addition_eval.txt \
  --out-dir runs/addition_v5 \
  --steps 2000 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --early-stop-patience 8 \
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
  --depth-penalty 0.01 \
  --consolidate-active-cells
```

## Continual update with autonomous growth

```bash
python train.py \
  --resume runs/addition_v5/checkpoint.pt \
  --train-file data/multiplication_train.txt \
  --eval-file data/multiplication_eval.txt \
  --retention-file data/addition_eval.txt \
  --out-dir runs/multiplication_v5 \
  --steps 1200 \
  --batch-size 32 \
  --seq-len 16 \
  --embedding-lr 1e-6 \
  --attention-lr 2e-6 \
  --ffn-lr 5e-6 \
  --other-lr 1e-6 \
  --cell-lr 5e-4 \
  --halt-lr 1e-4 \
  --consolidated-cell-scale 0.01 \
  --retention-replay-weight 1.0 \
  --retention-output-penalty 10.0 \
  --plastic-sparsity-weight 0.01 \
  --depth-penalty 0.01 \
  --enable-cell-growth \
  --cell-growth-patience 40 \
  --cell-growth-count 2 \
  --cell-growth-coverage-floor 0.75 \
  --enable-micro-growth \
  --micro-growth-patience 30 \
  --micro-growth-count 1 \
  --micro-growth-saturation 0.80 \
  --consolidate-active-cells
```

## Inspect inference

```bash
python chat.py --checkpoint runs/multiplication_v5/checkpoint.pt --show-routing
```

Telemetry reports adaptive depth, halting probabilities, population coverage, active cells, and internal micro-neuron saturation.

V1-V4 checkpoints are intentionally incompatible with V5. Retrain the base task once because V5 replaces fixed-depth blocks with a weight-tied recurrent block and changes cell internals.
