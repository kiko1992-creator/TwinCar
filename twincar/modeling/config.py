import os

# Data paths
DATA_ROOT = "/content/stanford_cars"
TRAIN_LABELS = "/content/train_labels.csv"
VAL_LABELS = "/content/val_labels.csv"
TEST_LABELS = "/content/test_labels.csv"

TRAIN_ROOT = os.path.join(DATA_ROOT, "cars_train/cars_train")
TEST_ROOT = os.path.join(DATA_ROOT, "cars_test/cars_test")

# Training hyperparameters
BATCH_SIZE = 32
VAL_RATIO = 0.1
RANDOM_SEED = 42
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 7

# Model/Save paths
BEST_MODEL_PATH = "/content/drive/MyDrive/efficientnetv2_best_model.pth"
METRICS_CSV_PATH = "/content/drive/MyDrive/metrics_log.csv"
