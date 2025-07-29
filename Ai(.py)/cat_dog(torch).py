import numpy
import pandas

!pip install gdown unzip

!gdown 1fuFurVV8rcrVTAFPjhQvzGLNdnTi1jWZ

!unzip -q /content/CATS_DOGS.zip

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

files[0] + files[2][0]

root , dires , files = os.walk("/content/CATS_DOGS/train")
class CatDogDataset(Dataset):
  def __init__(self , files : list[str]):
    super().__init__()
    set.files = files
  def __getitem__(self , idx):
    img = self.files[0] + self.files[-1][idx]
    img = Image.open(img)
    transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ])
    img = transform(img)

    return img
  def __len__(self):
    return (self.files[-1])

  dz = CatDogDataset(file = files)
  print(len(dz))
  print(dz[0])

