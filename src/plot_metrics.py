import json
import os

import matplotlib.pyplot as plt

CLASS_NAMES = ["pet", "background", "boundary"]
MODELS = [("unet", "U-Net"), ("vit", "ViT")]
COLORS = {"unet": "#1f77b4", "vit": "#d62728"}


def load(name):
    path = f"results_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    runs = {name: load(name) for name, _ in MODELS}
    runs = {k: v for k, v in runs.items() if v}

    if not runs:
        raise SystemExit("No results_*.json found. Run train.py first.")

    os.makedirs("assets", exist_ok=True)

    # Loss curves and mIoU, both models on shared axes.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for name, label in MODELS:
        history = runs.get(name)
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        color = COLORS[name]
        ax1.plot(epochs, [h["train_loss"] for h in history],
                 marker="o", color=color, label=f"{label} train")
        ax1.plot(epochs, [h["val_loss"] for h in history],
                 marker="o", color=color, linestyle="--", label=f"{label} val")
        ax2.plot(epochs, [h["miou"] for h in history],
                 marker="o", color=color, label=label)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("mIoU")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/metrics.png", dpi=120, bbox_inches="tight")
    print("saved assets/metrics.png")

    # Per-class IoU at the final epoch, grouped bars.
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    positions = range(len(CLASS_NAMES))

    for offset, (name, label) in enumerate(MODELS):
        history = runs.get(name)
        if not history:
            continue
        final = history[-1]["iou"]
        values = [0 if v is None else v for v in final]
        shifted = [p + offset * width for p in positions]
        ax.bar(shifted, values, width, label=label, color=COLORS[name])

    ax.set_xticks([p + width / 2 for p in positions])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU (final epoch)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("assets/per_class_iou.png", dpi=120, bbox_inches="tight")
    print("saved assets/per_class_iou.png")


if __name__ == "__main__":
    main()