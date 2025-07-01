import torch
from torchvision import models, transforms
from PIL import Image
from twincar.config import WEIGHTS_PATH, NUM_CLASSES

def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)

def predict(img_path, class_names):
    model = load_model()
    inp = preprocess(img_path)
    with torch.no_grad():
        logits = model(inp)
        idx = logits.argmax(dim=1).item()
    return class_names[idx]
