from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TRAIN_CSV = ROOT / "reports" / "train_split.csv"
VAL_CSV = ROOT / "reports" / "val_split.csv"
TEST_CSV = ROOT / "reports" / "test_split.csv"
TRAIN_IMG_DIR = ROOT / "cars_train"
TEST_IMG_DIR = ROOT / "cars_test"
WEIGHTS_PATH = ROOT / "models" / "resnet50_finetuned.pth"
NUM_CLASSES = 196
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 20
SEED = 42
