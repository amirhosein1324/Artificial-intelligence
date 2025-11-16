import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((227, 227)),
                                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])
                                           ]))

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((227, 227)),
                                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])
                                           ]))

plt.imshow(train_dataset[0][0][0], cmap="gray")

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
      super().__init__()
      self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, stride=4, kernel_size=11)
      self.batch1 = torch.nn.BatchNorm2d(96)
      self.pooling1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

      self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, stride=1, padding=2, kernel_size=5)
      self.batch2 = torch.nn.BatchNorm2d(256)
      self.pooling2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

      self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, stride=1, padding=1, kernel_size=3)
      self.batch3 = torch.nn.BatchNorm2d(384)
      self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, stride=1, padding=1, kernel_size=3)
      self.batch4 = torch.nn.BatchNorm2d(384)
      self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, stride=1, padding=1, kernel_size=3)
      self.batch5 = torch.nn.BatchNorm2d(256)
      self.pooling3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

      self.dp1 = torch.nn.Dropout(0.5)
      self.fc1 = torch.nn.Linear(in_features=9216, out_features=4096)
      self.dp2 = torch.nn.Dropout(0.5)
      self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
      self.fc3 = torch.nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, input):

      out_conv1 = self.conv1(input)
      out_batch1 = torch.nn.functional.relu(self.batch1(out_conv1))
      out_pooling1 = self.pooling1(out_batch1)

      out_conv2 = self.conv2(out_pooling1)
      out_batch2 = torch.nn.functional.relu(self.batch2(out_conv2))
      out_pooling2 = self.pooling2(out_batch2)

      out_conv3 = self.conv3(out_pooling2)
      out_batch3 = torch.nn.functional.relu(self.batch3(out_conv3))
      out_conv4 = self.conv4(out_batch3)
      out_batch4 = torch.nn.functional.relu(self.batch4(out_conv4))
      out_conv5 = self.conv5(out_batch4)
      out_batch5 = torch.nn.functional.relu(self.batch5(out_conv5))
      out_pooling3 = self.pooling3(out_batch5)

      flatten = out_pooling3.view(out_pooling3.shape[0], -1)
      out_dp1 = self.dp1(flatten)
      out_fc1 = torch.nn.functional.relu(self.fc1(out_dp1))
      out_dp2 = self.dp2(out_fc1)
      out_fc2 = torch.nn.functional.relu(self.fc2(out_dp2))
      out = self.fc3(out_fc2)
      return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on: {device}")

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)
criterion = torch.nn.CrossEntropyLoss()

epochs_num = 20
learning_rate = 1e-4
total_step = len(train_data_loader)
model = AlexNet(num_classes=10).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
print(120 * '-')



for epoch in range(epochs_num):
  for i, (images, labels) in enumerate(tqdm(train_data_loader)):
    images = images.to(device=device)
    labels = labels.to(device=device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  print(f"Epoch [{epoch+1}/{epochs_num}], Step [{i+1}/{total_step}], Loss: {loss.item()}")


  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      del images, labels, outputs

    print(f"Accuracy of the network on the {5000} validation images: {100*correct/total}%")


  print(120 * '-')

