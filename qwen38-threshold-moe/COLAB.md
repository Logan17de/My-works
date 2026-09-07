# Colab test note

The full 80-layer / 500-expert model is not a normal Colab-sized run. Colab is for validating **the exact training logic** on a scaled model first: router thresholding, layer boundary recompute, expert sliding cache, CPU Adafactor, dataset streaming, logs, checkpoint/resume.

## 1. Clone the research branch

```python
!git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
%cd My-works/qwen38-threshold-moe
```

## 2. Install only the Python dependencies

Colab already provides PyTorch/CUDA, so do not reinstall torch unless you have a specific reason.

```python
!pip install -q -r requirements.txt
```

Restart the runtime if Transformers was already imported before the upgrade.

## 3. Optional Hugging Face token

If the K2 aggregate dataset is gated/authenticated:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
```

The public smoke config uses `IFM/TxT360-v2` and does not intentionally depend on the gated aggregate.

## 4. Inspect without allocating all routed experts

```python
!python inspect_model.py --config configs/colab_smoke.json
```

## 5. Probe streaming data

```python
!python data_probe.py --config configs/colab_smoke.json --count 2
```

To test the exact K2 aggregate dataset ID:

```python
!python data_probe.py --config configs/full_g4.json --count 1
```

## 6. Run the smoke training

```python
!python train.py --config configs/colab_smoke.json
```

You will get a single live line similar to:

```text
step  3 | loss 8.1234 | ppl ... | tok/s ... | lr ... | route-gap ... | blocked ... | VRAM ... | cache ...
```

Persistent run outputs:

```text
runs/colab-smoke/
  metrics.csv
  loss.png
  throughput.png
  router_balance.png
  resolved_config.json
  expert_store/
  checkpoints/
```

## 7. Resume after interrupt

```python
!python train.py --config configs/colab_smoke.json --resume auto \
  --set training.max_steps=10
```

SIGINT/SIGTERM is handled by finishing the current layer-wise optimizer step and then checkpointing.

## 8. Change anything from CLI

No source edits are required for normal experiments:

```python
!python train.py --config configs/colab_smoke.json \
  --set model.num_layers=8 \
  --set model.num_experts=64 \
  --set data.sequence_length=8192 \
  --set training.expert_cache_layers=4 \
  --set training.layer_superbatch=1 \
  --set threshold.warmup_tokens=1000000 \
  --set training.max_steps=20 \
  --set training.run_name=8l-64e-test
```

Switch router control experiment:

```python
!python train.py --config configs/colab_smoke.json \
  --set model.router_activation=softmax
```

Change how many complete expert layers stay in VRAM:

```python
!python train.py --config configs/research_8x64.json \
  --set training.expert_cache_layers=1

!python train.py --config configs/research_8x64.json \
  --set training.expert_cache_layers=5
```

The model architecture is identical in those two runs; only transfer/cache behavior changes.

## 9. Before the 63B run

Run the `research_8x64.json` case and compare against a conventional small-model autograd reference. The acceptance criterion is gradient/loss agreement first, throughput second. Only then move to `configs/full_g4.json`.
