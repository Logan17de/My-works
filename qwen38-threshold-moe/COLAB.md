# Colab run note

## 1. Clone + install

```bash
!rm -rf /content/My-works
!git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
%cd /content/My-works/qwen38-threshold-moe
!pip install -r requirements.txt
```

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

The full config keeps the live expert masters and Adafactor state on fast Colab local storage at `/content/qtm-work`.

`--drive-root` is the persistent destination only. At every configured checkpoint (`save_every_steps`), clean interruption, and completion, the local work tree is synchronized to:

```text
/content/drive/MyDrive/qwen38-threshold-moe/work-backup
```

CSV logs, graphs, and resident checkpoints are also written automatically under the Drive root.

## 3. Validate once

```bash
!python train.py validate-layerwise --config configs/full_g4.json
```

## 4. Full 80-layer / 500-expert training

```bash
!python train.py train --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe
```

`full_g4.json` automatically uses as many routed-expert layers in VRAM as safely fit while reserving 20 GiB for the resident model and training transients. The selected cache size is printed at startup.

Training automatically produces:

```text
metrics.csv
training.png
routing.png
checkpoints/
work-backup/
```

No separate plotting or save command is required.

## 5. Resume from the latest synchronized checkpoint

```bash
!python train.py train --config configs/full_g4.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --resume latest
```

On a fresh Colab runtime, resume first restores the synchronized expert/optimizer work from Drive to local storage and then loads the matching checkpoint.
