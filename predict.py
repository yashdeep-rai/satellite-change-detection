
import torch
import cv2
from model import ChangeDetector

model = ChangeDetector()
model.load_state_dict(torch.load("change_model.pth"))

model.eval()

imgA = cv2.imread("testA.png")
imgB = cv2.imread("testB.png")

imgA = cv2.resize(imgA,(256,256))
imgB = cv2.resize(imgB,(256,256))

imgA = torch.tensor(imgA).permute(2,0,1).unsqueeze(0).float()/255
imgB = torch.tensor(imgB).permute(2,0,1).unsqueeze(0).float()/255

with torch.no_grad():
    pred = model(imgA,imgB)

mask = pred.squeeze().numpy()

cv2.imwrite("change_map.png",mask*255)

print("Change map saved.")
