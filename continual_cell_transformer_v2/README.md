# Continual Cell Transformer V2

V2 fixes the failure found in V1: newly allocated cells no longer enter the same
top-k competition as old cells.

## Architectural change

```text
attention
   ↓
stable bank (fixed top-k, preserved V1 path)
   ↓
plastic bank (independent top-k, allocated once)
   ↓
small plastic-only adapter + relative-match gate
   ↓
FFN
```

The stable bank is evaluated exactly as before. Plastic cells read the
stable-enriched hidden state, but cannot displace stable cells from routing. The
plastic gate compares plastic-bank similarity against stable-bank similarity, so
a plastic residual is favored only when the new bank matches the state better.
Only the plastic bank, its adapter and gate are trainable during continual
learning.

## Colab setup

```bash
!git pull origin agent/continual-cell-transformer
%cd /content/My-works/continual_cell_transformer_v2
!pip install -r requirements.txt
!python -m py_compile config.py tokenizer.py model.py train.py chat.py inspect_routes.py
```

## Import the trained V1 addition checkpoint and learn multiplication

Run this from `continual_cell_transformer_v2/`:

```bash
python train.py \
  --resume-v1 ../continual_cell_transformer/runs/addition/checkpoint.pt \
  --mode continual \
  --train-file ../continual_cell_transformer/data/multiplication_train.txt \
  --eval-file ../continual_cell_transformer/data/multiplication_eval.txt \
  --retention-file ../continual_cell_transformer/data/addition_eval.txt \
  --out-dir runs/multiplication_v2 \
  --allocate-plastic-cells 8 \
  --steps 1000 \
  --batch-size 32 \
  --seq-len 16 \
  --eval-interval 50 \
  --early-stop-patience 6
```

## Chat

```bash
python chat.py --checkpoint runs/multiplication_v2/best_checkpoint.pt
```

## Compare routing

```bash
python inspect_routes.py \
  --checkpoint runs/multiplication_v2/best_checkpoint.pt \
  --file ../continual_cell_transformer/data/addition_eval.txt \
  --file ../continual_cell_transformer/data/multiplication_eval.txt
```

Look for:

- the same stable cell IDs on addition before and after training;
- plastic gate near zero for addition;
- a higher plastic gate and consistent plastic cells for multiplication;
- multiplication loss falling while addition retention delta stays near zero.

This remains an experiment. Independent routing removes one interference path;
it does not guarantee that the learned plastic residual will never affect old
inputs.
