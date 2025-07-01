from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
import matplotlib.pyplot as plt

def gradcam_explain(model, val_dataset, class_names, device):
    model.eval()
    target_layer = model.blocks[-1] if hasattr(model, "blocks") else model.layer4[-1]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    # ...rest as in last_model.py
