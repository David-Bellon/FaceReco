import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm
import pickle
from model_arq import Model
import os

to_tensor = ToTensor()
to_pil = ToPILImage()
df = pd.read_csv("real_train.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):

    def __len__(self):
        return len(df)

    def __getitem__(self, index):
        image = Image.open("images/" + df.iloc[index]["image"])
        image = to_tensor(image)
        labels = torch.tensor([df.iloc[index]["tlc"], df.iloc[index]["brc"], df.iloc[index]["tlr"], df.iloc[index]["brr"]], dtype=torch.float)
        return image, labels

train_data = DataLoader(
    TrainDataset(),
    batch_size=500,
    shuffle=True
)

with open("train.obj", "rb") as file:
    df_train = pickle.load(file)

model = Model().to(device)
loss_function = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.01)

def train(input, real):
  optim.zero_grad()

  output = model(input)
  lost = loss_function(output, real)
  
  lost.backward()
  optim.step()

  return lost

history_loss = []
epochs = 150
for epoch in range(epochs):
  epoch_loss = 0
  if epoch > 0 and epoch % 45 == 0:
    for param_group in optim.param_groups:
      param_group["lr"] = param_group["lr"] * 0.5
      print(param_group["lr"])
  for i in tqdm(range(len(df_train)), position=0, leave=True):
    input = df_train.iloc[i]["image"].to(device)
    real = df_train.iloc[i]["label"].to(device)
    epoch_loss = epoch_loss + train(input, real)
  print(f"Epoch:{epoch}. Loss:{epoch_loss.item()/i}")
  history_loss.append(epoch_loss.item()/i)

torch.save(model, "Face_RecognitionV2.plt")