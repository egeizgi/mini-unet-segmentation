import os

import matplotlib.pyplot as plt
import torch

from data import get_loaders
from model import MiniUNet
from vit import ViTSegmenter

NUM_CLASSES = 3
SAMPLE_IDS = [12, 47, 103, 256]   # fixed for reproducibility


def load_model(name, device):
    """Return the trained model, or None if its checkpoint is missing."""
    path = f"model_{name}.pth"
    if not os.path.exists(path):
        print(f"skipping {name}: {path} not found")
        return None

    model = MiniUNet(num_classes=NUM_CLASSES) if name == "unet" \
        else ViTSegmenter(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [(label, load_model(name, device))
              for name, label in [("unet", "U-Net"), ("vit", "ViT")]]
    models = [(label, m) for label, m in models if m is not None]

    if not models:
        raise SystemExit("No checkpoints found. Run train.py first.")

    _, val_loader = get_loaders()
    val_ds = val_loader.dataset

    os.makedirs("assets", exist_ok=True)

    columns = ["Input", "Ground truth"] + [label for label, _ in models]
    fig, axes = plt.subplots(
        len(SAMPLE_IDS), len(columns),
        figsize=(3 * len(columns), 3 * len(SAMPLE_IDS)),
    )

    with torch.no_grad():
        for row, idx in enumerate(SAMPLE_IDS):
            img, mask = val_ds[idx]
            mask = mask.squeeze(0).long() - 1

            axes[row, 0].imshow(img.permute(1, 2, 0))
            axes[row, 1].imshow(mask, cmap="tab10", vmin=0, vmax=2)

            batch = img.unsqueeze(0).to(device)
            for col, (_, model) in enumerate(models, start=2):
                pred = model(batch).argmax(dim=1)[0].cpu()
                axes[row, col].imshow(pred, cmap="tab10", vmin=0, vmax=2)

            for col in range(len(columns)):
                axes[row, col].axis("off")

    for col, name in enumerate(columns):
        axes[0, col].set_title(name)

    plt.tight_layout()
    plt.savefig("assets/predictions.png", dpi=120, bbox_inches="tight")
    print("saved assets/predictions.png")


if __name__ == "__main__":
    main()