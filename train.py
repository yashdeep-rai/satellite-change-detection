import torch
from torch.utils.data import DataLoader
from dataset import ChangeDataset
from model import ChangeDetector

DATASET_PATH = "dataset"

dataset = ChangeDataset(DATASET_PATH)

loader = DataLoader(dataset,batch_size=4,shuffle=True)

model = ChangeDetector()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for imgA,imgB,mask in loader:

        optimizer.zero_grad()

        pred = model(imgA,imgB)

        loss = criterion(pred,mask)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

torch.save(model.state_dict(),"change_model.pth")

print("Training complete. Model saved.")
