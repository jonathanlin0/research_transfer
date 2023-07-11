import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from time import sleep

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

raw_data = unpickle("cifar_10_data/data_batch_1")
data_tensor = torch.from_numpy(raw_data[b"data"]).float()
data_tensor = torch.reshape(data_tensor, (10000, 3, 32, 32))

labels = unpickle("cifar_10_data/batches.meta")[b"label_names"]

img = data_tensor[4,:,:,:]
print(img.shape)

import torchvision.models as models


# Load weights from saved checkpoint file
# resnet_18 = models.resnet50(pretrained=False)
# checkpoint = torch.load('resnet18-f37072fd.pth')
# resnet_18.load_state_dict(checkpoint)
# resnet_18.eval()
# output = resnet_18(img)
# print(output)

# model = models.resnet50(pretrained=True)
# model.eval()
# with torch.no_grad:
#     output = model(img)


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
