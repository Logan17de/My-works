# Continual Cell Transformer

A small research prototype combining:

- a causal Transformer with RoPE;
- recurrent sparse concept-cell pools after selected attention layers;
- dormant reserve cells that can be recruited during continual training;
- row-wise plasticity masks so new cells learn faster than mature cells;
- an append-only hybrid byte tokenizer, so new tokens never renumber old tokens;
- separate learning rates for embeddings, attention, FFNs and concept cells;
- optional old-corpus retention measurement.

This is **not proven human-like learning**. It is a controlled implementation of
localized plasticity, expandable reserve capacity and stable old token IDs.

## Install

```bash
pip install torch
```

## 1. Initial training

```bash
python train.py \
  --train-file data/base.txt \
  --out-dir runs/base \
  --auto-add-words \
  --steps 2000 \
  --seal-active-cells
```

## 2. Introduce a new concept/token

```bash
python train.py \
  --resume runs/base/checkpoint.pt \
  --train-file data/new_chemistry.txt \
  --retention-file data/base_eval.txt \
  --out-dir runs/chemistry_update \
  --add-token "photoredox catalysis" \
  --auto-add-words \
  --enable-growth \
  --steps 800
```

What remains stable:

- all previous tokenizer IDs;
- all previous embedding and output rows at resize time;
- mature concept-cell rows receive only a small gradient;
- old-to-old recurrent connections are not changed when cells are recruited.

What can still forget:

- attention, FFNs, embeddings and mature cells are only *slowed*, not immutable;
- a new token changes segmentation wherever its exact string appears;
- resetting optimizer moments after vocabulary resize can alter optimization;
- reserve capacity is finite (`max_cells`).

## 3. Chat

```bash
python chat.py --checkpoint runs/chemistry_update/checkpoint.pt
```

## Important experiment

Always use a held-out old corpus:

```bash
--retention-file data/base_eval.txt
```

Compare retention loss before and after each new concept. Without that test, a
claim of continual learning is just optimism wearing a lab coat.
