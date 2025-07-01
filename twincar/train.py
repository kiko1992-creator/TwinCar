import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from twincar.config import *
from twincar.dataset import StanfordCarsDataset
from twincar.features import get_transforms

class CarClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = StanfordCarsDataset(TRAIN_CSV, TRAIN_IMG_DIR, get_transforms(train=True))
    val_ds = StanfordCarsDataset(VAL_CSV, TRAIN_IMG_DIR, get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = CarClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=LR)

    best_val_acc = 0
    train_hist, val_hist = {"loss": [], "acc": []}, {"loss": [], "acc": []}
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_hist["loss"].append(train_loss)
        train_hist["acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(y).sum().item()
                val_total += y.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_hist["loss"].append(val_loss)
        val_hist["acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print("Training complete. Best Val Acc: {:.4f}".format(best_val_acc))
