# Colab run note — FP8 full-resident

## 1. Clone + install

```bash
!rm -rf /content/My-works
!git clone -b moe-threshold-pretrain https://github.com/Logan17de/My-works.git
%cd /content/My-works/qwen38-threshold-moe
!pip install -r requirements-fp8.txt
```

## 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Train

```bash
!python train_fp8.py --config configs/full_g4_fp8.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe
```

The 80×500 routed expert bank stays resident on GPU in FP8. BF16 optimizer masters and Adafactor state stay on `/content/qtm-work`; Drive is synchronized only at checkpoints, clean interruption, or completion. CSV and graphs are automatic.

## 4. Resume

```bash
!python train_fp8.py --config configs/full_g4_fp8.json --drive-root /content/drive/MyDrive/qwen38-threshold-moe --resume latest
```
