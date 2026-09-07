# Qwen3.8-inspired Threshold MoE Pretraining Lab

A CLI-first research implementation of the architecture discussed in the continual-learning / expert-flow experiment:

**80 layers · hidden 256 · 500 routed experts/layer · 4 always-active shared experts · Top-4 routed · 8x expert expansion · Qwen3.8-style 3:1 GDN/attention + four-branch gated residual.**

The full model is ~63B parameters because capacity lives almost entirely in tiny experts. It is designed to explore whether a narrow backbone can acquire large, separable capacity through expert routing, and whether deliberately forcing broad expert utilization during pretraining creates a better substrate for continual/domain learning.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the locked design and [K2_RESEARCH.md](K2_RESEARCH.md) for the K2 Horizon data findings.

## What is implemented

- 500 routed experts per layer, packed as SwiGLU matrices
- 4 shared experts always active + Top-4 routed experts
- custom sigmoid router by default; softmax control available
- threshold curriculum after user-supplied `N` trained tokens
- Qwen3.8/Qwen4-Exp Gated DeltaNet from Transformers
- Qwen-style gated full attention every fourth layer for initial base pretraining
- Qwen3.8-style four-branch Gated Residual stream, scaled to hidden 256
- untied Qwen3.8 tokenizer embedding and LM head
- no n-gram embedding or MTP in the initial pretraining run
- 8K context default
- layer-wise forward checkpoints + reverse recomputation/backprop
- BF16 authoritative routed-expert masters in CPU RAM or mmap storage
- sliding whole-layer GPU expert cache (`--set training.expert_cache_layers=N`)
- momentum-free CPU Adafactor for routed experts
- current-layer gradient transfer as int8 / BF16 / FP32
- streaming Hugging Face or local data
- direct K2 Horizon dataset alias + public TxT360-v2 fallback
- one-line live training status
- CSV metrics and automatic loss/throughput/router graphs
- periodic checkpoints, clean SIGINT/SIGTERM save, and `--resume auto`
- every experiment knob overridable with `--set dotted.path=value`

## Quick start

```bash
git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
cd My-works/qwen38-threshold-moe
pip install -r requirements.txt

python inspect_model.py --config configs/colab_smoke.json
python data_probe.py --config configs/colab_smoke.json
python train.py --config configs/colab_smoke.json
```

Colab-specific cells are in [COLAB.md](COLAB.md).

## Full architecture inspection

This does not allocate the 63B expert bank:

```bash
python inspect_model.py --config configs/full_g4.json
```

Key arithmetic:

```text
one expert:       3 * 256 * 2048 = 1,572,864 params
routed experts:   80 * 500 * 1,572,864 = 62,914,560,000
shared experts:   80 * 4 * 1,572,864   =    503,316,480
```

## Give the threshold warmup N at launch

The full config intentionally uses `warmup_tokens=-1`, so thresholding cannot silently start at an invented N.

```bash
python train.py --config configs/full_g4.json \
  --set threshold.warmup_tokens=1000000000
```

At the default 500 experts, a 30% block fraction means 150 most-used experts are temporarily unavailable in an imbalanced layer. The router keeps scanning its ranking until four available experts are chosen.

## Sliding expert cache

Use spare VRAM by keeping more entire routed-expert banks resident:

```bash
# minimal VRAM, maximal transfers
python train.py --config configs/full_g4.json \
  --set training.expert_cache_layers=1

# keep five complete expert layers and slide
python train.py --config configs/full_g4.json \
  --set training.expert_cache_layers=5
```

This is placement, not an architecture change.

## Data CLI

Aliases:

```text
k2-horizon      -> IFM/K2-Horizon-Pretrain-Data
k2-txt360-v2    -> IFM/TxT360-v2
txt360          -> LLM360/TxT360
hf:ORG/NAME     -> arbitrary Hugging Face dataset
local:/path     -> local JSONL/text data
```

Examples:

```bash
# Exact K2 aggregate dataset when accessible
HF_TOKEN=... python data_probe.py --config configs/full_g4.json

# Public K2 source available for immediate plumbing tests
python data_probe.py --config configs/colab_smoke.json \
  --set data.source=k2-txt360-v2 \
  --set data.config_name=web-high-nltk-qa

# Any dataset
python train.py --config configs/colab_smoke.json \
  --set data.source=hf:some-org/some-dataset \
  --set data.config_name=some-config \
  --set data.text_field=text
```

The packer concatenates documents with EOS and emits fixed `sequence_length+1` token windows for ordinary next-token prediction. Dataset state stores the number of consumed examples and the partially packed token buffer for resume.

## CLI-only experimentation

```bash
python train.py --config configs/research_8x64.json \
  --set model.num_experts=96 \
  --set model.router_activation=sigmoid \
  --set threshold.imbalance_threshold=0.25 \
  --set threshold.block_fraction=0.20 \
  --set training.gradient_transfer_dtype=int8 \
  --set training.expert_cache_layers=6 \
  --set training.save_every_steps=25 \
  --set training.run_name=my-test
```

Print the fully resolved config without training:

```bash
python train.py --config configs/full_g4.json \
  --set threshold.warmup_tokens=1000000000 \
  --print-config
```

## Logging

Each run writes:

```text
runs/<run-name>/metrics.csv
runs/<run-name>/loss.png
runs/<run-name>/throughput.png
runs/<run-name>/router_balance.png
```

The terminal uses a single updating line rather than flooding Colab output.

## Checkpoint / interruption behavior

```bash
python train.py --config configs/colab_smoke.json \
  --set training.save_every_steps=10

python train.py --config configs/colab_smoke.json --resume auto
```

The signal handler finishes the current global layer-wise step, then saves model/optimizer/router/data/RNG state. With `expert_store=mmap`, the BF16 expert files themselves are live persistent masters.

For a **fully versioned crash-consistent** routed-expert snapshot, enable:

```bash
--set training.save_expert_snapshot=true
```

On the 500x80 model that copies roughly 117 GiB of expert masters per checkpoint, so it is intentionally off by default. Without it, clean handled interruptions are recoverable, but a hard power/runtime loss in the middle of a step can leave the live mmap experts ahead of the last resident checkpoint.

## FP8 status

The memory/runtime architecture is ready for an FP8 GPU shadow, but the first correctness path deliberately stages experts in BF16. Direct FP8 autograd is rejected instead of pretending BF16 compute is FP8. After the layer-wise gradient test passes, the staging backend can be swapped for Transformer Engine FP8/MXFP8 while keeping the BF16 CPU masters and Adafactor state unchanged.

That is also why `full_g4.json` currently starts with:

```json
"expert_compute_dtype": "bf16"
```

The research sequence should be **correct BF16 streamed trainer -> gradient equivalence -> FP8 shadow optimization**, not the reverse.
