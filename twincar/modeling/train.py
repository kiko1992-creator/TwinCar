import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss,
    cohen_kappa_score, matthews_corrcoef, jaccard_score
)
import timm
import seaborn as sns
import matplotlib.pyplot as plt
from twincar.dataset import StanfordCarsFromCSV, train_transform, val_transform
from twincar.config import *

# Data split (assuming you've already created train/val/test CSVs)

# Load datasets
df_all = pd.read_csv(TRAIN_LABELS)
df_train, df_val = train_test_split(
    df_all, test_size=VAL_RATIO, stratify=df_all['label'], random_state=RANDOM_SEED
)
df_train.to_csv('train_split.csv', index=False)
df_val.to_csv('val_split.csv', index=False)

train_dataset = StanfordCarsFromCSV(TRAIN_ROOT, 'train_split.csv', train_transform)
val_dataset = StanfordCarsFromCSV(TRAIN_ROOT, 'val_split.csv', val_transform)

labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

# Model setup
NUM_CLASSES = len(np.unique(labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.3)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# Training loop (same as your last_model.py)
EPOCHS = 25
patience, counter = 7, 0
best_val_f1 = 0
metrics_dict = { ... } # Use the same as in your script

# (Add the training loop as in last_model.py, logging metrics, early stopping, etc.)

# Save metrics and plots (see last_model.py for exact code)
