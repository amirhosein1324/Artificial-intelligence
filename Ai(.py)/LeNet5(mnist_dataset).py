import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

train_dataset[0][0].shape

train_dataset[0][1]

type(train_dataset[0][0])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((32, 32)),
                                               torchvision.transforms.Normalize(mean = 0.5 , std = 0.5)
                                           ]))

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((32, 32)),
                                               torchvision.transforms.Normalize(mean = 0.5 , std = 0.5)
                                           ]))

train_dataset[0][0].shape

train_dataset[0][1]

type(train_dataset[0][0])

plt.imshow(train_dataset[0][0][0], cmap="gray")

class Lenet5(torch.nn.Module):
  def __init__(self):
    super(Lenet5 , self ).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=1 , out_channels=6 , stride=1 , padding=0, kernel_size=5)
    self.sub2 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv3 = torch.nn.Conv2d(in_channels=6, out_channels=16, stride=1, padding=0, kernel_size=5)
    self.sub4 = torch.nn.MaxPool2d(kernel_size=2)
    self.fc5 = torch.nn.Linear(400 , 120)
    self.fc6 = torch.nn.Linear(120 , 84)
    self.out = torch.nn.Linear(84 , 10)
    self.rl = torch.nn.ReLU()
    self.sf = torch.nn.Softmax()

  def forward(self , x):
    x = self.conv1(x)
    x = self.sub2(x)
    x = self.conv3(x)
    x = self.sub4(x)
    x = x.view(x.shape[0] , -1)
    # print(view.shape)
    x = self.fc5(x)
    x = self.fc6(x)
    x = self.out(x)
    x = self.rl(x)
    x = self.sf(x)
    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_data_loader = DataLoader(train_dataset , batch_size = 64)
test_data_loader = DataLoader(test_dataset , batch_size = 64)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
model = Lenet5()
model = model.to(device=device)
optim = torch.optim.Adam(model.parameters() , lr = learning_rate)



for epoch in tqdm(range(10)):
  train_losses = []
  test_losses = []
  train_acc = 0
  test_acc = 0
  train_cnt = 0
  test_cnt = 0
  for train_data, train_label in train_data_loader:
    train_data = train_data.to(device=device)
    train_label = train_label.to(device=device)
    train_output = model(train_data)

    train_y_hat = torch.divide(train_output, train_output.sum(axis=1, keepdims=True))
    train_y_hat_indices = torch.argmax(train_y_hat, axis=1)
    train_loss = loss_fn(train_output, train_label)

    optim.zero_grad()
    train_loss.backward()
    optim.step()
    train_losses.append(train_loss.item())


  with torch.no_grad():
    for test_data, test_label in test_data_loader:
      test_data = test_data.to(device=device)
      test_label = test_label.to(device=device)
      test_output = model(test_data)

      test_y_hat = torch.divide(test_output, test_output.sum(axis=1, keepdims=True))
      test_y_hat_indices = torch.argmax(test_y_hat, axis=1)
      test_loss = loss_fn(test_output, test_label)
      test_losses.append(test_loss.item())

  train_acc += sum(train_label == train_y_hat_indices)
  train_cnt += len(train_label)
  test_acc += sum(test_label == test_y_hat_indices)
  test_cnt += len(test_label)

  print(f"epoch: {epoch}")
  print(f"\ttrain_accuracy: {train_acc / train_cnt}")
  print(f"\ttrain_loss: {sum(train_losses) / len(train_losses)}")
  print(f"\ttest_accuracy: {test_acc / test_cnt}")
  print(f"\ttest_loss: {sum(test_losses) / len(test_losses)}")







