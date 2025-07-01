import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss, cohen_kappa_score,
    matthews_corrcoef, jaccard_score
)
import torch.nn.functional as F
import shutil

from twincar.dataset import StanfordCarsFromCSV, train_transform, test_transform
from twincar.config import device, NUM_CLASSES, NUM_EPOCHS, patience, BATCH_SIZE

# Update these paths as needed
extract_dir = '/content/stanford_cars'
train_root = f"{extract_dir}/cars_train/cars_train"
test_root = f"{extract_dir}/cars_test/cars_test"

# Data splits (should be created with the main notebook or scripts)
train_csv = '/content/train_split.csv'
val_csv = '/content/val_split.csv'

train_dataset = StanfordCarsFromCSV(train_root, train_csv, train_transform)
val_dataset = StanfordCarsFromCSV(train_root, val_csv, test_transform)

# Weighted sampling for class balance
labels = [label for _, label in train_dataset]
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(device)

optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

metrics_dict = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_precision_macro': [], 'val_precision_weighted': [],
    'val_recall_macro': [], 'val_recall_weighted': [],
    'val_f1_macro': [], 'val_f1_weighted': [],
    'val_hamming': [], 'val_cohen_kappa': [],
    'val_mcc': [], 'val_jaccard_macro': [],
    'val_top3': [], 'val_top5': [],
}

best_val_f1 = 0
counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / total
    train_acc = correct / total
    metrics_dict['train_loss'].append(train_loss)
    metrics_dict['train_acc'].append(train_acc)

    # --- VALIDATION ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_probs, val_preds, val_targets = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item() * imgs.size(0)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_probs.extend(probs.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    val_loss /= val_total
    val_acc = val_correct / val_total
    val_preds_np = np.array(val_preds)
    val_targets_np = np.array(val_targets)
    val_probs_np = np.array(val_probs)

    # Standard metrics
    val_precision_macro = precision_score(val_targets_np, val_preds_np, average='macro', zero_division=0)
    val_precision_weighted = precision_score(val_targets_np, val_preds_np, average='weighted', zero_division=0)
    val_recall_macro = recall_score(val_targets_np, val_preds_np, average='macro', zero_division=0)
    val_recall_weighted = recall_score(val_targets_np, val_preds_np, average='weighted', zero_division=0)
    val_f1_macro = f1_score(val_targets_np, val_preds_np, average='macro', zero_division=0)
    val_f1_weighted = f1_score(val_targets_np, val_preds_np, average='weighted', zero_division=0)
    top3_acc = np.mean([
        label in np.argsort(prob)[-3:] for prob, label in zip(val_probs_np, val_targets_np)
    ])
    top5_acc = np.mean([
        label in np.argsort(prob)[-5:] for prob, label in zip(val_probs_np, val_targets_np)
    ])
    val_hamming = hamming_loss(val_targets_np, val_preds_np)
    val_cohen_kappa = cohen_kappa_score(val_targets_np, val_preds_np)
    val_mcc = matthews_corrcoef(val_targets_np, val_preds_np)
    val_jaccard_macro = jaccard_score(val_targets_np, val_preds_np, average='macro', zero_division=0)

    metrics_dict['val_loss'].append(val_loss)
    metrics_dict['val_acc'].append(val_acc)
    metrics_dict['val_precision_macro'].append(val_precision_macro)
    metrics_dict['val_precision_weighted'].append(val_precision_weighted)
    metrics_dict['val_recall_macro'].append(val_recall_macro)
    metrics_dict['val_recall_weighted'].append(val_recall_weighted)
    metrics_dict['val_f1_macro'].append(val_f1_macro)
    metrics_dict['val_f1_weighted'].append(val_f1_weighted)
    metrics_dict['val_hamming'].append(val_hamming)
    metrics_dict['val_cohen_kappa'].append(val_cohen_kappa)
    metrics_dict['val_mcc'].append(val_mcc)
    metrics_dict['val_jaccard_macro'].append(val_jaccard_macro)
    metrics_dict['val_top3'].append(top3_acc)
    metrics_dict['val_top5'].append(top5_acc)

    scheduler.step(val_f1_macro)

    print(f"Epoch {epoch+1:2d} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"F1(macro): {val_f1_macro:.4f} | Top3: {top3_acc:.3f} | Top5: {top5_acc:.3f}")

    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        torch.save(model.state_dict(), 'twin_car_best_model_v2.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

shutil.copy('twin_car_best_model_v2.pth', '/content/drive/MyDrive/twin_car_best_model_v2.pth')
print("✅ Best model saved")
