import cv2
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

def detect_objects(model, frame):
    img_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    return predictions
