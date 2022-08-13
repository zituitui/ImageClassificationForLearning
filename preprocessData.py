import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST"
    ,train=True
    ,download=True
    ,transform=transforms.Compose([transforms.ToTensor()])
)

train_set_loader = torch.utils.data.DataLoader(train_set)
print(train_set_loader)
print("load data finished")