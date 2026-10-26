from model import MiniUNet
from data import get_loaders
from metrics import count_intersection_union, mean_iou, iou_from_counts
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_loaders(batch_size=8)

    model = MiniUNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCH = 5

    history = []
    for epoch in range(EPOCH):
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

        num_classes = 3
        total_inter = [0] * num_classes
        total_union = [0] * num_classes

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device) - 1

                logits = model(imgs)
                val_losses.append(criterion(logits, masks).item())

                pred = logits.argmax(dim=1)

                inter, union = count_intersection_union(pred, masks, num_classes)
                for c in range(num_classes):
                    total_inter[c] += inter[c]
                    total_union[c] += union[c]

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        ious = iou_from_counts(total_inter, total_union)

        print(f"epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f}")

        for c, iou in enumerate(ious):
            print(f"sınıf {c}: IoU {iou:.4f}" if iou is not None else f"sınıf {c}: yok")
        print(f"mIoU: {mean_iou(ious):.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "iou": ious,
            "miou": mean_iou(ious),
        })

    Path("assets").mkdir(exist_ok=True)
    with open("results.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), "model.pth")
    print("saved model.pth and results.json")

if __name__ == '__main__':
    main()