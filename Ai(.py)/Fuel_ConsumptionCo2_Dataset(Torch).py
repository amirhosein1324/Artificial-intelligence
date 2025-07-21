import numpy as np
import torch
import math
import pandas as pd

data = pd.read_csv("FuelConsumptionCo2.csv")
data.head

X = torch.tensor(data["ENGINESIZE"].values, dtype=torch.float32)
X2 = torch.tensor(data["FUELCONSUMPTION_COMB"].values, dtype=torch.float32)
Y = torch.tensor(data["CO2EMISSIONS"].values, dtype=torch.float32)
X = (X - X.mean()) / X.std()
X2 = (X2 - X2.mean()) / X2.std()
Y = (Y - Y.mean()) / Y.std()
inputs = torch.stack([X, X2], dim=1)
class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=torch.nn.Linear(2,3)
    self.fc2=torch.nn.Linear(3,4)
    self.fc3=torch.nn.Linear(4,1)
    self.flatten=torch.nn.Flatten(0,1)
  def forward(self,input):
    first=self.fc1(input)
    middle=self.fc2(first)
    out=self.fc3(middle)
    out=self.flatten(out)
    return out
model=MyModel()
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate=1e-6
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
for t in range(1000):
    y_pred = model(inputs)
    loss = loss_fn(y_pred, Y)
    if t % 100 == 99:
      print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(list(model.parameters()))

