
import torch
from torch.utils.data import DataLoader
from dataset import ChangeDataset
from model import ChangeDetector
from metrics import iou_score
from download_dataset import download_dataset
from tqdm import tqdm

download_dataset()

DATASET_PATH = "dataset"

dataset = ChangeDataset(DATASET_PATH)

loader = DataLoader(dataset,batch_size=4,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChangeDetector().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0
    total_iou = 0

    for imgA,imgB,mask in tqdm(loader):

        imgA = imgA.to(device)
        imgB = imgB.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        pred = model(imgA,imgB)

        loss = criterion(pred,mask)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        total_iou += iou_score(pred,mask)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("Loss:", total_loss/len(loader))
    print("IoU:", total_iou/len(loader))

torch.save(model.state_dict(),"change_model.pth")

print("Training complete.")
