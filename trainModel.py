import torch
import torchvision.utils

import preprocessData
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import Network
import torch.optim as optim

torch.set_printoptions(120)
torch.set_grad_enabled(True)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


train_set = preprocessData.train_set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

network = Network.Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch
        predicts = network(images)
        loss = F.cross_entropy(predicts, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(predicts, labels)

    print("epoch: ", epoch
          , "total_correct: ", total_correct,
          "loss: ", total_loss)

torch.save(network.state_dict(),"./models/Trained_model")