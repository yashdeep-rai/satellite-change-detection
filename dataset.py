
import os
import cv2
import torch
from torch.utils.data import Dataset

class ChangeDataset(Dataset):

    def __init__(self, root_dir):

        self.root = root_dir

        self.A = sorted(os.listdir(os.path.join(root_dir, "A")))
        self.B = sorted(os.listdir(os.path.join(root_dir, "B")))
        self.label = sorted(os.listdir(os.path.join(root_dir, "label")))

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):

        imgA_path = os.path.join(self.root, "A", self.A[idx])
        imgB_path = os.path.join(self.root, "B", self.B[idx])
        label_path = os.path.join(self.root, "label", self.label[idx])

        imgA = cv2.imread(imgA_path)
        imgB = cv2.imread(imgB_path)
        mask = cv2.imread(label_path, 0)

        imgA = cv2.resize(imgA, (256,256))
        imgB = cv2.resize(imgB, (256,256))
        mask = cv2.resize(mask, (256,256))

        imgA = torch.tensor(imgA).permute(2,0,1).float()/255
        imgB = torch.tensor(imgB).permute(2,0,1).float()/255
        mask = torch.tensor(mask).unsqueeze(0).float()/255

        return imgA, imgB, mask
