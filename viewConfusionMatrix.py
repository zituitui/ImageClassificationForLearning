import torch
import torchvision.utils

import preprocessData
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import Network
import torch.optim as optim
import matplotlib.pyplot as plt
train_set = preprocessData.train_set

from sklearn.metrics import confusion_matrix

def get_all_predicts(model, loader):
    all_preds = torch.tensor([])
    for batch in train_loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds
cmt = torch.load("./Tensors/ConfusionMatrix.pt")
network = Network.Network()
network.load_state_dict(torch.load("./models/Trained_model"))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
network.eval()

train_preds = get_all_predicts(network, train_loader)
cm = confusion_matrix(train_set.targets,train_preds.argmax(dim=1))
print(cm)
print(cmt)