import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import csv
import numpy as np

from sort import data_helper
from models import *

### TRAINING
DH = data_helper()
DH.load_training_data("train_temp.csv")
DH.load_testing_data("test_temp.csv")

train_size = int(DH.training_size * 0.8)
dev_size = DH.training_size - train_size
print(train_size, dev_size)

train_pixels = torch.FloatTensor(DH.training_set[:train_size])
train_labels = torch.LongTensor(DH.training_labels[:train_size])
train_data_set = TensorDataset(train_pixels, train_labels)
train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle=True)

dev_pixels = torch.FloatTensor(DH.training_set[train_size:])
dev_labels = torch.LongTensor(DH.training_labels[train_size:])
dev_data_set = TensorDataset(dev_pixels, dev_labels)
dev_data_loader = DataLoader(dev_data_set, batch_size=64, shuffle=False)

test_pixels = torch.FloatTensor(DH.test_set)
test_loader = DataLoader(test_pixels, batch_size=32, shuffle=False)

print(len(test_pixels))
model = CNN()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
loss = nn.NLLLoss()

for epoch in range(30):
    epoch_loss = 0
    train_accuracy = [0,0]

    for pixels, labels in train_data_loader:
        y_pred = model(pixels)
        predictions = torch.argmax(y_pred, dim=1)
        output = loss(y_pred, labels)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        epoch_loss += output
        train_accuracy = [train_accuracy[0] + (predictions == labels).sum(), train_accuracy[1] + labels.shape[0]]

    dev_epoch_loss = 0
    dev_accuracy = [0,0]

    for pixels_d, labels_d in dev_data_loader:
        y_pred_d = model(pixels_d)
        predictions_d = torch.argmax(y_pred_d, dim=1)
        output_d = loss(y_pred_d, labels_d)
        dev_epoch_loss += output_d
        dev_accuracy = [dev_accuracy[0] + (predictions_d == labels_d).sum(), dev_accuracy[1] + labels_d.shape[0]]
    print(f"{epoch}: {round(float(epoch_loss/train_size),5)}, {round(int(train_accuracy[0])/int(train_accuracy[1]),5)}, {round(float(dev_epoch_loss/dev_size),5)}, {round(int(dev_accuracy[0])/int(dev_accuracy[1]),5)}")

    """
    #DEBUG PROBABILITIES SCRIPT
    max_probabilities = torch.max(torch.exp(y_pred_d), dim=1)[0]
    correct = max_probabilities*(predictions_d == labels_d)
    incorrect = max_probabilities*(~(predictions_d == labels_d))
    correct_clone = correct.clone()
    correct_clone[correct_clone != 0] = 1
    print(torch.sum(correct)/torch.sum(correct_clone))
    print(torch.sum(incorrect)/(len(incorrect)-torch.sum(correct_clone)))
    input("click enter to continue \n")
    """

prediction_array = np.zeros(len(test_pixels))
current_index = 0

test_size = len(test_pixels)
for pixels, i in zip(test_loader, range(test_size)):
    y_pred = model(pixels)
    predictions = torch.argmax(y_pred, dim=1).detach()
    prediction_array[current_index:current_index+len(y_pred)] = predictions
    current_index += len(y_pred)

prediction_array = [[j, int(i)] for i,j in zip(prediction_array, range(1,test_size+1))]

prediction_array = [["ImageId","Label"]] + prediction_array

with open('file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(prediction_array)

"""
### EVALUATION
for pixels, i in zip(test_loader, range(len(test_pixels))):
    y_pred = model(pixels)
    print(torch.argmax(y_pred, dim=1))
    DH.visualise_test_data(i)
"""
