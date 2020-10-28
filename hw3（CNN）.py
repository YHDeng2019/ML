import torch
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import time
import random


# 定义一个读取图片的函数readfile()
def readfile(path, only_data):
    # os.listdir()返回一个目录下的文件和文件夹名的list
    numofpic, pics = len(os.listdir(path)), os.listdir(path)
    # 数据
    x = np.zeros((numofpic, 128, 128, 3), dtype=np.uint8)
    # 标签
    y = np.zeros(numofpic)
    for (i, filename) in enumerate(pics, start=0):
        img = cv2.imread(path + '/' + filename)
        # resize每个图片为128*128的大小
        imggg = cv2.resize(img, (128, 128))
        x[i] = imggg
    if only_data:
        return x
    else:
        for (i, filename) in enumerate(pics, start=0):
            # 从文件名中获取图片标签
            y[i] = eval(filename.split('_')[0])
        return x, y


# 数据增强
trian_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


#  定义Dataset类 必须重写init\len\getitem三个方法
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        xx = self.x[index]
        if self.transform is not None:
            xx = self.transform(xx)
        if self.y is not None:
            yy = self.y[index]
            return xx, yy
        else:
            return xx


# 读取数据
pic_path = 'CNN/data/food-11'
train_x, train_y = readfile(pic_path + '/training', False)
val_x, val_y = readfile(pic_path + '/validation', False)
test_data = readfile(pic_path + '/testing', True)
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)

print("训练集的大小为：", len(train_x))
print("验证集的大小为：", len(val_x))
print("测试集的大小为：", len(test_data))

# random.seed(1)
batch_size = 128
train_set = ImgDataset(train_x, train_y, trian_transforms)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = ImgDataset(val_x, val_y, trian_transforms)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
train_val_set = ImgDataset(train_val_x, train_val_y, trian_transforms)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride=kernel_size, padding)
        # input 维度 [3, 128, 128]

        # 卷积层
        # Sequential它是一个顺序容器, 其中模块的添加顺序与在构造函数中传递模块时的顺序相同。
        # BatchNorm应作用在非线性映射前
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64,128,128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 64,64,64

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 128,64,64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 128,32,32

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256,32,32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 256,16,16

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512,16,16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 512,8,8

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512,8,8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 512,4,4
        )

        # 全连接层，输入为卷积层输出降维之后的一维数据
        self.fullyConnect = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        # 用view函数将512*4*4降维成1维 和reshape一样
        dimension_reduction_out = cnn_out.view(cnn_out.size(0), -1)
        fc_out = self.fullyConnect(dimension_reduction_out)
        return fc_out


# gpu = torch.device('cuda:0')
# cpu = torch.device('cpu')
# model = Net().to(gpu)
model = Net().cuda()
loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optim = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
epoch_num = 30  # 迭代次数
for epoch in range(epoch_num):
    star_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    '''
    model.train()是确保 model 是在训练模式(开启 Dropout 等...)
    如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
    其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
    而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    '''
    model.train()
    for i, data in enumerate(train_val_loader):
        # 清空梯度，每进来一个batch计算一次梯度，更新一次网络
        optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零
        model_pred = model(data[0].cuda()) # 利用 model 得到预测的概率分布，这边实际上是调用模型的 forward 函数
        batch_loss = loss(model_pred, data[1].cuda())
        batch_loss.backward()
        # step()让优化器进行单次优化，一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数
        optimizer.step()
        # 统计分类正确的个数  argmax()用于返回一个numpy数组中最大值的索引值,axis=1比较行，axis=0比较列
        train_acc += np.sum(np.argmax(model_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    # 验证集
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 将结果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, epoch_num, time.time() - star_time,
               train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

# 测试集，生成文件
test_set = ImgDataset(x=test_data, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model.eval()
pred_list = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for j in test_label:
            pred_list.append(j)
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, label in enumerate(pred_list):
        f.write('{},{}\n'.format(i, label))

# plt.figure()
# plt.plot(plt_train_loss)
# plt.plot(plt_val_loss)
# plt.title("Loss")
#
# plt.figure()
# plt.plot(plt_train_acc)
# plt.plot(plt_val_acc)
# plt.title("Accuracy")
# plt.show()
