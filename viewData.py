import torch
import torchvision.utils

import preprocessData
import numpy as np
import matplotlib.pyplot as plt

train_set = preprocessData.train_set
train_loader = preprocessData.train_set_loader

torch.set_printoptions(linewidth=120)
print(len(train_set))
print(train_set.train_labels)
print(train_set.train_labels.bincount())

#
sample = next(iter(train_set))
print(len(sample))
print(type(sample))
image, label = sample

#plt.imshow(image.squeeze(),cmap='gray')
#plt.show()
#print("label:",label)

batch = next(iter(train_loader))
print(len(batch))
print(type(batch))
images, labels = batch

grid = torchvision.utils.make_grid(images,nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid,(1,2,0)))
plt.show()

