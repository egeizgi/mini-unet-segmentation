import json

import matplotlib.pyplot as plt

CLASS_NAMES = ["pet", "background", "boundary"]


def main():
    with open("results.json") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for c, name in enumerate(CLASS_NAMES):
        ax2.plot(epochs, [h["iou"][c] for h in history], marker="o", label=name)
    ax2.plot(epochs, [h["miou"] for h in history], marker="s", linestyle="--", label="mIoU")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/metrics.png", dpi=120, bbox_inches="tight")
    print("saved assets/metrics.png")


if __name__ == "__main__":
    main()