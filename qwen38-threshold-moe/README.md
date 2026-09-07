# Qwen3.8-inspired Threshold MoE pretraining experiment

Single-GPU research implementation for the 80-layer, hidden-256, 500-routed-expert architecture discussed in the design thread.

## Current architecture

- 80 layers, hidden 256
- 500 routed experts/layer
- four shared experts/layer
- softmax + Top-4 router only
- threshold-routing curriculum after configurable warmup
- Qwen-style 3 Gated DeltaNet : 1 gated full-attention layout
- four-branch Gated Residual
- Qwen partial RoPE (`theta=1e7`) in full-attention layers
- expert SwiGLU `256 -> 2048 -> 256`
- Qwen3.8 tokenizer, 248,320 vocabulary, untied embedding/head
- initial 8K pretraining; no n-gram embedding or MTP

## Memory strategy

Routed expert masters live in BF16 CPU mmap/RAM on the fast local work disk. A sliding window of complete routed-expert layers is copied to GPU. Forward stores only layer-boundary activations on CPU. Backward recomputes one layer at a time, computes `dX`, transfers routed expert gradients to CPU (INT8 by default), updates expert masters with factored Adafactor, and updates resident parameters with AdamW.

The full config uses `expert_cache_layers: "auto"`. At startup it measures free VRAM and fills it with as many complete expert layers as fit while retaining the configured safety reserve (`gpu_cache_reserve_gib`, 20 GiB in `full_g4.json`). A manual `--cache-layers N` still overrides this.

The last forward cache window is retained into reverse backward, so those layers are not reloaded before backward starts.

## CLI

Available commands include `show-config`, `info`, `inspect`, `probe-data`, `init-experts`, `validate-layerwise`, `train`, and `plot`. Important experiment settings have direct flags such as `--save-every`, `--resume`, `--drive-root`, `--work-dir`, `--cache-layers`, `--superbatch`, `--seq-len`, learning-rate controls, and threshold controls. Remaining fields can use repeatable `--set dotted.key=value` overrides.

## Colab / Drive persistence

For the full Colab/G4 configuration:

```text
local hot work: /content/qtm-work/<run-name>/
persistent logs/checkpoints: <drive-root>/runs/<run-name>/
persistent expert/optimizer mirror: <drive-root>/work-backup/<run-name>/
```

`--drive-root` no longer makes Google Drive the live mmap disk. Training reads and writes expert masters plus Adafactor state locally. At every configured checkpoint, clean interruption, and completion, the local work tree is synchronized to the persistent `work-backup` directory. A fresh-runtime `--resume latest` restores that synchronized work back to local storage and verifies its saved step matches the resident checkpoint before resuming.

The full routed expert masters are about 117 GiB, so the Colab local disk and persistent destination both need enough capacity.

## Live logging / recovery

The live terminal line reports the signals needed to diagnose this experiment:

- current layer / total layers
- GPU expert-cache capacity
- expert layers/GiB loaded during the transition and load time
- layer compute time
- MoE execution backend (`native-grouped`, Hugging Face grouped fallback, or eager fallback)
- current-layer expert coverage, hottest expert, routing gap, blocked experts, and reroute percentage
- end-of-step loss, throughput, mean/min expert coverage, mean/max routing gap, blocked layers/experts, and reroute percentage

Before a complete optimizer step exists, throughput is shown as unavailable rather than a misleading `0 tok/s`.

Training automatically writes:

- `metrics.csv`
- `training.png`
- `routing.png`
- periodic resident checkpoints
- synchronized expert/optimizer work at checkpoint time

SIGINT/SIGTERM requests finish the current optimizer step and then checkpoint + synchronize. A hard runtime loss cannot execute cleanup, so recovery is from the most recent completed synchronization.

## Dataset streaming

The full configuration uses the released public K2-oriented repositories through `configs/k2_public_mix.json`. The exact original K2 Horizon mixture weights are not public, so this fallback mixture is explicitly not claimed to reproduce IFM's original data schedule. Tokenized documents are streamed and packed continuously into fixed next-token sequences.

See `K2_RESEARCH.md` and `COLAB.md` for details.

## Important validation boundary

Run `validate-layerwise` before scaling. The memory-saving schedule assumes recomputation produces the same `dX` and parameter gradients as ordinary autograd before a layer's parameters are updated.
