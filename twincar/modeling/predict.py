import torch
import pandas as pd
from torch.utils.data import DataLoader
from twincar.dataset import StanfordCarsFromCSV, val_transform
from twincar.config import *
import timm

# Load model
NUM_CLASSES = ... # Set this based on your classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# Validation dataset
val_dataset = StanfordCarsFromCSV(TRAIN_ROOT, VAL_LABELS, val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Make predictions and calculate metrics (copy from your script)
