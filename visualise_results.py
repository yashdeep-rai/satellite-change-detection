import os
import random
import torch
import cv2
import numpy as np
from model import ChangeDetector

DATASET_PATH = "dataset"
OUTPUT_DIR = "sample_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# pick random image
files = os.listdir(os.path.join(DATASET_PATH, "A"))
img_name = random.choice(files)

imgA_path = os.path.join(DATASET_PATH, "A", img_name)
imgB_path = os.path.join(DATASET_PATH, "B", img_name)

print("Using image:", img_name)

# load model
model = ChangeDetector()
model.load_state_dict(torch.load("change_model.pth"))
model.eval()

# load images
imgA = cv2.imread(imgA_path)
imgB = cv2.imread(imgB_path)

imgA_resized = cv2.resize(imgA, (256,256))
imgB_resized = cv2.resize(imgB, (256,256))

imgA_tensor = torch.tensor(imgA_resized).permute(2,0,1).unsqueeze(0).float()/255
imgB_tensor = torch.tensor(imgB_resized).permute(2,0,1).unsqueeze(0).float()/255

with torch.no_grad():
    pred = model(imgA_tensor, imgB_tensor)

mask = pred.squeeze().numpy()

# threshold mask
mask = (mask > 0.5).astype(np.uint8)

# create overlay
overlay = imgB_resized.copy()

overlay[mask == 1] = [0,0,255]  # red highlight

# blend
alpha = 0.6
result = cv2.addWeighted(imgB_resized, 1-alpha, overlay, alpha, 0)

# save results
cv2.imwrite(os.path.join(OUTPUT_DIR, "before.png"), imgA)
cv2.imwrite(os.path.join(OUTPUT_DIR, "after.png"), imgB)
cv2.imwrite(os.path.join(OUTPUT_DIR, "change_overlay.png"), result)

print("Visualization saved in:", OUTPUT_DIR)