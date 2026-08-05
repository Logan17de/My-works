# Continual Cell Transformer

Research prototype combining a causal Transformer with sparse recurrent concept-cell pools, dormant reserve cells, local plasticity and an append-only tokenizer.

This does **not** prove human-like learning. The testable claim is narrower: new concepts should concentrate more change in newly recruited cells while old-task loss remains comparatively stable.

## Files

- `config.py` — architecture and plasticity settings
- `model.py` — RoPE Transformer, recurrent cells, allocation and vocabulary resizing
- `tokenizer.py` — UTF-8 byte fallback plus append-only concept tokens
- `train.py` — initial and continual training, growth and retention evaluation
- `chat.py` — interactive generation

## Initial training

```bash
python train.py \
  --train-file data/base.txt \
  --out-dir runs/base \
  --auto-add-words \
  --steps 2000 \
  --seal-active-cells
```

## Continual update with new concepts

```bash
python train.py \
  --resume runs/base/checkpoint.pt \
  --train-file data/new_concepts.txt \
  --retention-file data/base_eval.txt \
  --out-dir runs/update_1 \
  --add-token "photoredox catalysis" \
  --auto-add-words \
  --enable-growth \
  --steps 800
```

Old token IDs are preserved. The embedding and LM-head rows are copied exactly when the vocabulary grows. Optimizer moments are restarted after resizing because the parameter shapes change.

## Chat

```bash
python chat.py --checkpoint runs/update_1/checkpoint.pt
```

## Important limitation

The reserve pool has a fixed maximum allocation. Attention, FFN, embeddings and mature cells are slowed rather than perfectly immutable, so catastrophic forgetting remains possible. Always provide `--retention-file` and compare loss before and after each continual update.
