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

train_set = preprocessData.train_set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)


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

network = Network.Network()
network.load_state_dict(torch.load("./models/Trained_model"))
network.eval()

train_preds = get_all_predicts(network, train_loader)
print(train_preds.shape)
stacked = torch.stack(
    (
        train_set.targets
        , train_preds.argmax(dim=1)
    )
    , dim=1
)

cmt = torch.zeros(10, 10, dtype=torch.int32)
for p in stacked:
    j, k = p.tolist()
    cmt[j, k] = cmt[j, k] + 1
print(cmt)
torch.save(cmt, "./Tensors/ConfusionMatrix.pt")