# Mini U-Net - Oxford-IIIT Pet Segmentation

Semantic segmentation on the Oxford-IIIT Pet dataset, with the U-Net architecture and the mIoU metric implemented from scratch. No pretrained segmentation models and no metric libraries were used

## Results 

Test split (3,669 images), 5 epochs, 128×128 input, batch size 8, Adam with lr=1e-3, seed 42.

| Class | IoU |
|---|---|
| 0 — pet | 0.6361 |
| 1 — background | 0.7888 |
| 2 — boundary | 0.3035 |
| **mIoU** | **0.5761** |

Per-epoch progression:

| Epoch | Train loss | Val loss | mIoU |
|---|---|---|---|
| 1 | 0.7721 | 0.7082 | 0.3915 |
| 2 | 0.6672 | 0.6190 | 0.4766 |
| 3 | 0.5660 | 0.5418 | 0.5172 |
| 4 | 0.5140 | 0.4821 | 0.5819 |
| 5 | 0.4857 | 0.4761 | 0.5761 |

## Observations

- The boundry class cover only a thin 2, 3 pixel band. 
- For thin structures, a small spatial offset costs heavily in IoU: the intersection shrinks fast while the union stays large
- The boundary itself is ambiguous even in the ground truth.
- A wider receptive field does not help here, what is needed is high-resolution detail.

## Implemented from scratch

- `src/model.py` - U-Net architecture (encoder, decoder, bottleneck, skips).
- `src/metrics.py` - per-class IoU and mIoU. Intersection and union counts are accumulated across the full validation set and divided only at the end, rather than averaging per-batch IoU values.

## Setup

```bash
pip install -r requirements.txt
python src/train.py
```

The dataset is downloaded automatically on first run (~800 MB).

## Implementation notes

- Masks are resized with `NEAREST` interpolation. Bilinear interpolation would produce intermediate values between class indices, inventing labels that do not exist.
- Labels are shifted from 1–3 to 0–2, since `CrossEntropyLoss` expects zero-based class indices.
- Training and evaluation are seeded for reproducibility.

## Repository layout

```
src/
├── model.py      # MiniUNet
├── metrics.py    # IoU and mIoU
├── data.py       # transforms and dataloaders
└── train.py      # training and evaluation loop
```