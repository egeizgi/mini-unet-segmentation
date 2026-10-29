import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from model import MiniUNet
from data import get_loaders
from metrics import count_intersection_union, mean_iou, iou_from_counts
from vit import ViTSegmenter

num_classes = 3

def build_model(name):
    if name == 'unet':
        return MiniUNet(num_classes=num_classes)
    if name == 'vit':
        return ViTSegmenter(num_classes=num_classes)

def run(name, epochs=5, lr=1e-3, batch_size=8):
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_loaders(batch_size=batch_size)

    model = build_model(name=name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device) - 1

            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        total_inter = [0] * num_classes
        total_union = [0] * num_classes

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device) - 1

                logits = model(imgs)
                val_losses.append(criterion(logits, masks).item())
                pred = logits.argmax(dim=1)
                inter, union = count_intersection_union(pred, masks, num_classes)
                for c in range(num_classes):
                    total_inter[c] = inter[c]
                    total_union[c] = union[c]

        ious = iou_from_counts(total_inter, total_union)
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": sum(val_losses) / len(val_losses),
            "iou": ious,
            "miou": mean_iou(ious),
        }
        history.append(record)
        print(f"[{name}] epoch {record['epoch']}: "
              f"train {record['train_loss']:.4f} | "
              f"val {record['val_loss']:.4f} | "
              f"mIoU {record['miou']:.4f}")
    with open(f"results_{name}.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), f"model_{name}.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "vit"], default="unet")
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()
    run(args.model, epochs=args.epoch)

if __name__ == "__main__":
    main()