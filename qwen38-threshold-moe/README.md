# Qwen3.8-inspired Threshold MoE pretraining experiment

Single-GPU research implementation for the 80-layer, hidden-256, 500-routed-expert architecture.

## Architecture

- 80 layers, hidden 256
- 500 routed experts/layer + 4 shared experts/layer
- softmax Top-4 routed experts
- threshold-routing curriculum after configurable warmup
- Qwen-style 3 Gated DeltaNet : 1 gated full-attention layout
- four-branch Gated Residual
- partial RoPE (`theta=1e7`)
- expert SwiGLU `256 -> 2048 -> 256`
- Qwen3.8 tokenizer, vocab 248,320, untied embedding/head
- 8K initial pretraining; no n-gram/MTP initially

## FP8 full-resident training

The main G4/Colab path is now `train_fp8.py` + `configs/full_g4_fp8.json`.

All 80×500 routed expert shadows stay resident in GPU VRAM as E4M3 FP8. Only the current layer is materialized as a BF16 autograd work tensor. Expert GEMMs use TorchAO MXFP8 grouped matrix multiplication for differentiable FP8 forward/backward. The 8×8192-token layer superbatch is concatenated into one layer call so the grouped kernel sees 65,536 tokens instead of eight separate 8K calls.

The authoritative routed-expert optimizer masters remain BF16 mmap files on fast Colab local disk. Routed gradients are packed to INT8, CPU factored Adafactor updates the BF16 master, then that layer's FP8 GPU shadow is refreshed. This keeps sub-FP8 optimizer updates while eliminating expert-layer streaming during forward/backward.

The memory-critical routed bank is 62.91456B parameters = 58.59 GiB at one byte/parameter. The full model is about 63.62B parameters; the smaller non-routed model components stay BF16 for stability.

## Persistence

```text
local hot work: /content/qtm-work/<run-name>/
Drive logs/checkpoints: <drive-root>/runs/<run-name>/
Drive expert/optimizer mirror: <drive-root>/work-backup/<run-name>/
```

Drive is not used in the training hot path. At each configured checkpoint, clean interruption, and completion, the local BF16 expert masters + Adafactor state are synchronized to Drive. `--resume latest` restores that synchronized state on a fresh runtime.

The local routed BF16 masters are about 117.19 GiB and the factored Adafactor state is about 0.99 GiB, so allow roughly 119 GiB for the hot work tree and similar persistent backup capacity.

## Run

```bash
python train_fp8.py --config configs/full_g4_fp8.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe
```

Resume:

```bash
python train_fp8.py --config configs/full_g4_fp8.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --resume latest
```

Training automatically writes `metrics.csv`, `training.png`, `routing.png`, live expert-routing metrics, and periodic checkpoints.

## Dataset

The full config streams the public K2-oriented mixture in `configs/k2_public_mix.json`. Exact original K2 Horizon mixture weights are not public, so this is not claimed to reproduce IFM's original data schedule.

## Validation / current status

The earlier layer-wise recomputation validator matched direct autograd on the tested layer. The new TorchAO MXFP8 path is a prototype backend and must be runtime-validated on the actual Blackwell G4 environment; it intentionally fails loudly rather than silently falling back to BF16 expert execution.
