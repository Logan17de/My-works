# Colab run note

This note contains runner commands only. The model/trainer implementation stays in the repository.

## 1. Clone + install

```bash
!rm -rf /content/My-works
!git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
%cd /content/My-works/qwen38-threshold-moe
!pip install -r requirements.txt
```

## 2. Mount Drive

Use the notebook's **Mount Google Drive** cell. All commands below use:

```text
/content/drive/MyDrive/qwen38-threshold-moe
```

`--drive-root` stores both runs/checkpoints and mmap expert masters there, so resume survives a Colab disconnect.

## 3. Inspect + validate + probe data

```bash
!python train.py inspect --config configs/colab_smoke.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --save-every 1 --plot-every 1
!python train.py validate-layerwise --config configs/colab_smoke.json --device cuda
!python train.py probe-data --config configs/colab_smoke.json --samples 2
```

## 4. Smoke train

```bash
!python train.py train --config configs/colab_smoke.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name colab-smoke --save-every 1 --plot-every 1 --max-steps 5 --cache-layers 2
```

## 5. Resume

```bash
!python train.py train --config configs/colab_smoke.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name colab-smoke --resume latest --save-every 1 --max-steps 10 --cache-layers 2
```

## 6. Main direct CLI controls

```text
--save-every N
--plot-every N
--max-steps N
--resume latest|none|PATH
--drive-root PATH
--output-dir PATH
--work-dir PATH
--run-name NAME
--cache-layers N
--superbatch N
--microbatch N
--seq-len N
--dataset NAME
--dataset-config NAME
--shuffle-buffer N
--lr VALUE
--expert-lr VALUE
--expert-grad-dtype int8|bf16|fp32
--expert-compute-dtype bf16|fp16|fp32
--warmup-tokens N
--imbalance-threshold VALUE
--block-fraction VALUE
--threshold / --no-threshold
--snapshot-experts / --no-snapshot-experts
```

Every other config key is still available through repeatable `--set dotted.key=value`.

## 7. Full 80-layer / 500-expert run

Inspect first:

```bash
!python train.py inspect --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name full-g4 --save-every 10 --cache-layers 5 --max-steps 1000
```

Train:

```bash
!python train.py train --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name full-g4 --save-every 10 --plot-every 5 --cache-layers 5 --max-steps 1000 --warmup-tokens YOUR_N
```

Resume:

```bash
!python train.py train --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name full-g4 --resume latest --save-every 10 --plot-every 5 --cache-layers 5 --max-steps 1000 --warmup-tokens YOUR_N
```

The BF16 routed-expert masters for the full model are about 117 GiB, so confirm Drive capacity before using a Drive-backed full run.

## 8. Rebuild graphs

```bash
!python train.py plot --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --run-name full-g4
```
