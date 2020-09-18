import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import time

from sort import data_helper
from models import *

c = nn.Conv2d(1, 10, 5, 1)
d = nn.Conv2d(10, 20, 5, 1)

print(c.weight.shape)

x = torch.rand((100, 784))
x = torch.reshape(x,(-1, 1, 28, 28))
y = d(c(x))
print(y.shape)
print(torch.flatten(y, 1).shape)

"""
DH = data_helper()
DH.load_training_data("train_temp.csv")

train_pixels = torch.FloatTensor(DH.training_set)
train_labels = torch.LongTensor(DH.training_labels)
train_data_set = TensorDataset(train_pixels, train_labels)
train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=True)

model = CONV()

for pixels, labels in train_data_loader:
    y_pred = model(pixels)
    print(y_pred.shape)
    time.sleep(2)
"""
