import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import models
from twincar.dataset import StanfordCarsFromCSV, test_transform
from twincar.config import device, NUM_CLASSES, BATCH_SIZE

# Update paths as needed
extract_dir = '/content/stanford_cars'
test_root = f"{extract_dir}/cars_test/cars_test"
test_csv = '/content/test_labels.csv'

test_dataset = StanfordCarsFromCSV(test_root, test_csv, test_transform, has_labels=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load('twin_car_best_model_v2.pth', map_location=device))
model = model.to(device)
model.eval()

import scipy.io
meta = scipy.io.loadmat(f"{extract_dir}/car_devkit/devkit/cars_meta.mat")
class_names = [x[0] for x in meta['class_names'][0]]

preds, filenames = [], []
with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        filenames.extend(names)

predicted_names = [class_names[i] for i in preds]
results_df = pd.DataFrame({'filename': filenames, 'class_name': predicted_names})
results_df.to_csv('test_predictions_named.csv', index=False)
print("✅ test_predictions_named.csv saved.")
