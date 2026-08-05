# Continual Cell Transformer

Research prototype combining a causal Transformer with sparse recurrent concept-cell pools, append-only token insertion, local plasticity and expandable routing banks.

This does **not** prove human-like learning. The testable claim is narrower: a new task should learn through a newly allocated routing bank without displacing the routes used by older tasks.

## V2: stable routing banks

The first version placed every active cell in one global top-k competition. New cells could therefore displace old cells even when every old weight was frozen.

V2 changes that design:

- the original cells remain in bank `0`;
- each allocation event creates a separate bank;
- top-k routing happens independently inside each bank;
- new banks may read the detached output of older banks;
- every new bank owns a trainable output adapter;
- sealed banks can be completely frozen with `--mature-cell-scale 0`;
- automatic growth is capped by `--max-growth-events`;
- `--allocate-cells` is the preferred controlled experiment.

## Files

- `config.py` — architecture, routing-bank and plasticity settings
- `model.py` — RoPE Transformer, stable multi-bank routing and vocabulary resizing
- `tokenizer.py` — UTF-8 byte fallback plus append-only concept tokens
- `train.py` — initial/continual training, held-out evaluation and retention checks
- `chat.py` — deterministic interactive generation with optional routing telemetry

## Initial addition training

```bash
python train.py \
  --train-file data/addition_train.txt \
  --eval-file data/addition_eval.txt \
  --out-dir runs/addition_v2 \
  --steps 3000 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --early-stop-patience 8 \
  --d-model 128 \
  --layers 4 \
  --heads 4 \
  --d-ff 512 \
  --max-cells 256 \
  --initial-cells 64 \
  --top-k-cells 8 \
  --seal-active-cells
```

## Continual multiplication update

Use the clean addition checkpoint, not the failed single-bank multiplication checkpoint.

```bash
python train.py \
  --resume runs/addition_v2/checkpoint.pt \
  --train-file data/multiplication_train.txt \
  --eval-file data/multiplication_eval.txt \
  --retention-file data/addition_eval.txt \
  --out-dir runs/multiplication_v2 \
  --allocate-cells 8 \
  --freeze-backbone \
  --mature-cell-scale 0 \
  --cell-lr 5e-4 \
  --weight-decay 0 \
  --steps 1500 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --eval-batches 20 \
  --early-stop-patience 8 \
  --log-cell-routing \
  --seal-active-cells
```

Expected bank layout:

```text
routing banks: [{0: 64, 1: 8}, {0: 64, 1: 8}]
```

The trainer separately reports:

- old-task loss before allocation;
- old-task loss immediately after allocation;
- multiplication held-out loss during training;
- old-task loss after training;
- selected cell IDs inside each bank.

## Chat and inspect routing

```bash
python chat.py \
  --checkpoint runs/multiplication_v2/checkpoint.pt \
  --show-routing
```

## Important boundary

Separate banks preserve routing competition, but they do not guarantee that eight new cells have enough capacity to learn multiplication through a frozen backbone. If multiplication fails while addition remains stable, the next question is capacity or interface expressiveness—not catastrophic forgetting.
