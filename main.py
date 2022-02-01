from torch import optim
from data.mnist import MNIST, MNIST_NORMALIZATION
from data.emnist import EMNIST
from model.cnn import simple_cnn
from torchvision import datasets,transforms
import torch
import numpy as np
from maha import maha_distance

tf = transforms.Compose([
            transforms.ToTensor(),
            MNIST_NORMALIZATION,
        ])
train_data = MNIST('./dataset/',download=True,even=True,transform=tf)
test_data = MNIST('./dataset/',download=True,even=True,transform=tf,train=False)
near_ood = MNIST('./dataset/',download=True,even=False,transform=tf,train=False)
far_ood = EMNIST('./dataset/',split='letters',download= True,transform=tf,train=False)

train_loader = torch.utils.data.DataLoader(train_data,batch_size = 128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = 128,shuffle=True)
near_ood_loader = torch.utils.data.DataLoader(near_ood,batch_size = 128,shuffle=True)
far_ood_loader = torch.utils.data.DataLoader(far_ood,batch_size = 128,shuffle=True)

model = simple_cnn(y_dim=5).to('cuda')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-7)
maha = maha_distance(5)

for e in range(5):
    for image,label in train_loader:
        out = model(image.to('cuda'))
        # print(out.shape,label)
        loss = criterion(out,label.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        # break

ftrain = []
with torch.no_grad():
    for image,label in train_loader:
        feature = model.net(image.to('cuda'))
        ftrain += feature.cpu().numpy().tolist()
maha.model_feature(ftrain)

ftest= []
with torch.no_grad():
    for image,label in test_loader:
        feature = model.net(image.to('cuda'))
        ftest += feature.cpu().numpy().tolist()
test_score = maha.score(ftest)
print(np.mean(test_score))


ftest= []
with torch.no_grad():
    for image,label in near_ood_loader:
        feature = model.net(image.to('cuda'))
        ftest += feature.cpu().numpy().tolist()
test_score = maha.score(ftest)
print(np.mean(test_score))

ftest= []
with torch.no_grad():
    for image,label in far_ood_loader:
        feature = model.net(image.to('cuda'))
        ftest += feature.cpu().numpy().tolist()
test_score = maha.score(ftest)
print(np.mean(test_score))