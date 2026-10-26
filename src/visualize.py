import matplotlib.pyplot as plt
import torch

from data import get_loaders
from model import MiniUNet

SAMPLE_IDS = [12, 47, 103, 256]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniUNet(num_classes=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    _, val_loader = get_loaders()
    val_ds = val_loader.dataset

    fig, axes = plt.subplots(len(SAMPLE_IDS), 3, figsize=(9, 3 * len(SAMPLE_IDS)))

    with torch.no_grad():
        for row, idx in enumerate(SAMPLE_IDS):
            img, mask = val_ds[idx]
            mask = mask.squeeze(0).long() - 1

            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()

            axes[row, 0].imshow(img.permute(1, 2, 0))
            axes[row, 1].imshow(mask, cmap="tab10", vmin=0, vmax=2)
            axes[row, 2].imshow(pred, cmap="tab10", vmin=0, vmax=2)

            for col in range(3):
                axes[row, col].axis("off")

    for col, name in enumerate(["Input", "Ground truth", "Prediction"]):
        axes[0, col].set_title(name)

    plt.tight_layout()
    plt.savefig("assets/predictions.png", dpi=120, bbox_inches="tight")
    print("saved assets/predictions.png")


if __name__ == "__main__":
    main()