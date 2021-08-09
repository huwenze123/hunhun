import torch
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import time
from torch import nn
# 配置参数
#DOWNLOAD_CIFAR = True
batch_size = 32  # 每次喂入的数据量
lr = 0.01  # 学习率
step_size = 10  # 每n个epoch更新一次学习率
epoch_num = 50  # 总迭代次数
num_print = int(1152//batch_size//4)  #每n次batch打印一次

train_dir = 'VGG_Popcorn_Dataset/train'
train_datasets = torchvision.datasets.ImageFolder(train_dir, transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
 
test_dir = 'VGG_Popcorn_Dataset/test'
test_datasets = torchvision.datasets.ImageFolder(test_dir, transform=torchvision.transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)


# 按batch_size 打印出dataset里面一部分images和label
classes = ('healthy', 'big_leaf_spot', 'small_leaf_spot', 'rust')

def image_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


def label_show(loader):
    global classes
    dataiter = iter(loader)  # 迭代遍历图片
    images, labels = dataiter.__next__()
    image_show(make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    return images, labels

#VGG网络
#from torch import nn


class Vgg16Net(nn.Module):
    def __init__(self):
        super(Vgg16Net, self).__init__()

        # 第一层，2个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入3通道，卷积核3*3，输出64通道（如256*256*3的样本图片，(256+2*1-3)/1+1=256，输出256*256*64）
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输入64通道，卷积核3*3，输出64通道（输入256*256*64，卷积3*3*64*64，输出256*256*64）
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输入256*256*64，输出128*128*64
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入64通道，卷积核3*3，输出128通道（输入128*128*64，卷积3*3*64*128，输出128*128*128）
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3*3，输出128通道（输入128*128*128，卷积3*3*128*128，输出128*128*128）
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输入128*128*128，输出64*64*128
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入128通道，卷积核3*3，输出256通道（输入64*64*128，卷积3*3*128*256，输出64*64*256）
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3*3，输出256通道（输入64*64*256，卷积3*3*256*256，输出64*64*256）
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3*3，输出256通道（输入64*64*256，卷积3*3*256*256，输出64*64*256）
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输入64*64*256，输出32*32*256
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第四层，3个卷积层和1个最大池化层
        self.layer4 = nn.Sequential(
            # 输入256通道，卷积3*3，输出512通道（输入32*32*256，卷积3*3*256*512，输出32*32*512）
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入32*32*512，卷积3*3*512*512，输出32*32*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入32*32*512，卷积3*3*512*512，输出32*32*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入32*32*512，输出16*16*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第五层，3个卷积层和1个最大池化层
        self.layer5 = nn.Sequential(
            # 输入512通道，卷积3*3，输出512通道（输入16*16*512，卷积3*3*512*512，输出16*16*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入16*16*512，卷积3*3*512*512，输出16*16*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入512通道，卷积3*3，输出512通道（输入16*16*512，卷积3*3*512*512，输出16*16*512）
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输入16*16*512，输出8*8*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 32768)
        x = self.fc(x)
        return x

#label_show(train_dataloader)
print(torch.cuda.device_count())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = torch.load('popcorn_classification_model.pkl')  #加载模型
model = model.to(device)
start = time.time()
# test
model.eval()
correct = 0.0
total = 0
softmax_results = np.zeros((32,4))
with torch.no_grad():  # 测试集不需要反向传播
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        outputs = outputs.cuda().data.cpu().numpy()
        exponi = np.exp(outputs)
        summ = exponi.sum(axis=1)
        ###如果需要输出每种种类的softmax概率，请取消下面注释###
        '''
        for i in range(4):
            for j in range(32):
                softmax_results[j][i] = exponi[j][i]/summ[j]
        #print(outputs,exponi,softmax_results)
        print(softmax_results)
        '''
        print(outputs)
        total += inputs.size(0)
        correct += torch.eq(pred,labels).sum().item()
print('Accuracy of the network on the about 230 test images: %.2f %%' % (100.0 * correct / total))
end = time.time()
print("The test time is:",end-start)
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    c = (pred == labels.to(device)).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += float(c[i])
        class_total[label] += 1
#每个类的ACC
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
'''



