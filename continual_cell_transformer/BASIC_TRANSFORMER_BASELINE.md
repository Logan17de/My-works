# Basic Transformer Baseline

This baseline is the control experiment for the Continual Cell Transformer V7 arithmetic-generalization result.

## Architecture

The baseline deliberately removes all continual-cell mechanisms:

- no routed cells,
- no micro-neurons,
- no structural growth,
- no consolidation,
- no retention machinery,
- no adaptive halting.

It keeps the same core Transformer choices used by V7:

- causal self-attention with RoPE,
- RMSNorm,
- SwiGLU FFN,
- `d_model=128`,
- `n_heads=4`,
- `d_ff=512`,
- dropout `0.1`,
- the same `DynamicByteTokenizer`,
- the same `math_answer_eos_v2` objective,
- the same arithmetic datasets and exact generation-based evaluation.

The default baseline uses **8 ordinary untied Transformer blocks** because V7 can execute up to 8 recurrent passes. This is a compute-depth comparison, not a parameter-matched comparison: the baseline has a separate attention/FFN parameter set at every layer, while V7 reuses one recurrent block. The trainer prints the exact parameter count.

`--layers` is configurable, so a smaller parameter-control baseline can also be tested later without changing the implementation.

## Generate the same held-out-operand benchmark

If the files do not already exist:

```bash
python generate_math_generalization_datasets.py \
  --heldout-digit 7 \
  --repeats 50 \
  --seed 17
```

## Train the baseline

This matches the V7 generalization experiment's data, width, heads, FFN size, batch size, sequence length, objective, learning rate for ordinary Transformer parameters, mastery rule, and training budget.

```bash
rm -rf runs/basic_transformer_addition_generalization

python train_basic_transformer.py \
  --train-file data/addition_generalization_train.txt \
  --eval-file data/addition_seen_canonical_eval.txt \
  --out-dir runs/basic_transformer_addition_generalization \
  --steps 3000 \
  --batch-size 32 \
  --seq-len 32 \
  --eval-interval 50 \
  --eval-batches 20 \
  --log-interval 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.001 \
  --mastery-stop \
  --mastery-accuracy 1.0 \
  --mastery-patience 2 \
  --mastery-max-new-tokens 4 \
  --d-model 128 \
  --heads 4 \
  --d-ff 512 \
  --layers 8 \
  --dropout 0.1 \
  --lr 2e-4 \
  --weight-decay 0 \
  --grad-clip 1.0 \
  --seed 17
```

## Quick generalization test

```bash
python evaluate_basic_transformer_generalization.py \
  --checkpoint runs/basic_transformer_addition_generalization/checkpoint.pt \
  --operation addition \
  --heldout-digit 7 \
  --max-new-tokens 3 \
  --limit-per-split 25 \
  --seed 17
```

## Full generalization test

```bash
python evaluate_basic_transformer_generalization.py \
  --checkpoint runs/basic_transformer_addition_generalization/checkpoint.pt \
  --operation addition \
  --heldout-digit 7 \
  --max-new-tokens 3
```

The key comparison with V7 is:

| Split | V7 measured result | Basic Transformer |
|---|---:|---:|
| seen canonical | 100.00% | pending |
| seen unseen-format | 98.46% | pending |
| held-out operand 7 | 0.00% | pending |
| held-out + unseen format | 0.00% | pending |
| notation | 49.38% | pending |
| two-digit | 0.00% | pending |

No baseline result is claimed until the experiment is actually run.

## Interactive check

```bash
python chat_basic_transformer.py \
  --checkpoint runs/basic_transformer_addition_generalization/checkpoint.pt \
  --max-new-tokens 8
```
