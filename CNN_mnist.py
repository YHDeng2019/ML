import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.modules as nn
import time

traindata = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
testdata = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

#
# def get_x_y(data):
#     x = np.zeros((len(data), 1, 28, 28), dtype=np.uint8)
#     y = np.zeros(len(data), dtype=np.uint8)
#     for i, (dataline,target) in enumerate(data):
#         x[i] = dataline
#         y[i] = target
#     return x, y
#
#
# class imgDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = torch.Tensor(x)
#         self.y = torch.LongTensor(y)
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, index):
#         xx = self.x[index]
#         yy = self.y[index]
#         return xx, yy


batch_size = 128
# x, y = get_x_y(traindata)
# trainset = imgDataset(x, y)
trainset = traindata
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# x2, y2 = get_x_y(testdata)
# testset = imgDataset(x2, y2)
testset = testdata
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)


class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        input维度:[60000,1,28,28]
        '''
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # [16,28,28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16,14,14]

            nn.Conv2d(16, 32, 3, 1, 1),  # [32,14,14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32,7,7]

            nn.Conv2d(32, 64, 3, 1, 1),  # [64,7,7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1)  # [64,4,4]
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 0),  # [32,26,26]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 0),  # [64,24,24]
            nn.MaxPool2d(2, 2),  # [64,12,12]
            nn.Dropout2d(0.25),
            nn.Flatten(start_dim=1),  # 9216*1
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

        self.fullConnect = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.full2 = nn.Sequential(
            nn.Linear(28 * 28, 10),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.cnn2(x)
        return out  # size???


model = CnnNet().cuda()
loss = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(model.parameters(), lr=1e-3)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)
epoch_num = 30

for epoch in range(epoch_num):
    num_correct = 0.0
    train_loss = 0.0
    time0 = time.time()

    # 训练
    correct = 0
    model.train()
    for i, data in enumerate(train_loader):
        optim.zero_grad()
        output = model(data[0].cuda())  # type torch.float32 shape[128,10]
        batch_loss = loss(output, data[1].cuda())
        batch_loss.backward()
        optim.step()

        train_loss += batch_loss.item()  # type???
        _,predict = torch.max(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct += (predict == data[1].cuda()).sum()
        # correct += pred.eq(data[1].cuda().view_as(pred)).sum().item()
        # num_correct += np.sum(np.argmax(output.cpu().detach().numpy(), axis=1) == data[1].numpy())

    train_acc = correct.item() / trainset.__len__()

    #  测试
    test_loss = 0.0
    num_correct_test = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data[0].cuda())
            batch_loss = loss(test_pred, data[1].cuda())
            test_loss += batch_loss.item()

            num_correct_test += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

        test_acc = int(num_correct_test) / testset.__len__()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f Test Acc: %3.6f Loss: %3.6f' % \
              (epoch + 1, epoch_num, time.time() - time0, train_acc, train_loss / trainset.__len__(),
               test_acc, test_loss / testset.__len__()))
