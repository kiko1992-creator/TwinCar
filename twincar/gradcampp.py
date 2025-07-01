import torch
import numpy as np
import os
from torchvision import models
from twincar.dataset import StanfordCarsFromCSV, test_transform
from twincar.config import device, NUM_CLASSES

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Update these paths as needed
extract_dir = '/content/stanford_cars'
test_root = f"{extract_dir}/cars_test/cars_test"
test_csv = '/content/test_labels.csv'

test_dataset = StanfordCarsFromCSV(test_root, test_csv, test_transform, has_labels=False)

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

target_layer = model.layer4[-1]
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

os.makedirs('gradcam_outputs', exist_ok=True)
for i in range(5):  # Visualize first 5 test images
    try:
        image_tensor = test_dataset[i][0]
        input_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        pred_idx = outputs.argmax().item()
        targets = [ClassifierOutputTarget(pred_idx)]
        cam_input = input_tensor.requires_grad_()
        grayscale_cam = cam(input_tensor=cam_input, targets=targets)
        if grayscale_cam is None or grayscale_cam[0] is None:
            continue
        grayscale_cam = grayscale_cam[0]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        vis = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        filename = f"gradcam_outputs/campp_{i:02d}_class_{pred_idx}.png"
        Image.fromarray(vis).save(filename)
        print(f"Grad-CAM++ visualization saved: {filename}")
    except Exception as e:
        print(f"❌ Failed on image {i}: {e}")
