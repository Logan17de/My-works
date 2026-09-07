# Colab run note

This note contains commands only. The model/trainer implementation stays in the repository.

## 1. Clone the experiment branch

```bash
!git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
%cd My-works/qwen38-threshold-moe
```

## 2. Install dependencies

```bash
!pip install -r requirements.txt
```

Restart the Colab runtime once if pip asks for it, then return to the same directory.

## 3. Check GPU

```bash
!nvidia-smi
```

## 4. Show the resolved smoke configuration

```bash
!python train.py show-config --config configs/colab_smoke.json
```

## 5. Check model/memory numbers

```bash
!python train.py info --config configs/full_g4.json
```

## 6. Probe the public K2/TxT360 stream

```bash
!python train.py probe-data --config configs/colab_smoke.json --samples 3
```

## 7. Validate layer-wise recomputation before training

```bash
!python train.py validate-layerwise --config configs/colab_smoke.json --device cuda
```

## 8. Run the small Colab smoke training

```bash
!python train.py train --config configs/colab_smoke.json
```

## 9. Resume after interruption

```bash
!python train.py train --config configs/colab_smoke.json --set training.resume=latest
```

## 10. Run the 8-layer / 64-expert experiment

```bash
!python train.py train --config configs/research_8x64.json
```

## 11. Change anything from CLI only

```bash
!python train.py train --config configs/research_8x64.json --set model.num_experts=96 --set training.expert_cache_layers=4 --set training.layer_superbatch=4 --set data.sequence_length=4096 --set training.max_steps=200
```

## 12. Use the exact K2 aggregate if your Hugging Face account has access

```bash
!huggingface-cli login
!python train.py probe-data --config configs/full_g4.json --samples 3
```

## 13. Use another public K2 subset from CLI

```bash
!python train.py train --config configs/colab_smoke.json --set data.source=k2-code --set data.config_name=code-thinking-v1
```

## 14. Use a custom mixture file

```bash
!python train.py train --config configs/research_8x64.json --set data.mixture_file=configs/k2_public_mix.example.json
```

## 15. Generate/update the training graph

```bash
!python train.py plot --config configs/colab_smoke.json
```

## 16. Full 80-layer / 500-expert configuration

Do not start this on a normal Colab A100 until the smoke and 8x64 validation runs are clean. On the intended large-memory host, initialize/run it with:

```bash
!python train.py init-experts --config configs/full_g4.json
!python train.py train --config configs/full_g4.json --set threshold.warmup_tokens=YOUR_N
```

Training writes a one-line live status, `metrics.csv`, `training.png`, periodic checkpoints, and a clean interruption checkpoint.
