import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import shutil



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # fully connect
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    global arg
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--numepoches', type=int, default=1)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    arg = parser.parse_args()
    print(arg)

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)
    num = 0
    for batch_datas, batch_labels in trainloader:
        num = num + 1
        print('train_size',batch_datas.size(), batch_labels.size())
        if num > 10:
            break

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    num = 0
    for batch_datas, batch_labels in testloader:
        num = num + 1
        print('test_size',batch_datas.size(), batch_labels.size())
        if num > 10:
            break
    # Train the network
    model = MY_CNN(
        # Data Loader
        train_loader=trainloader,
        test_loader=testloader,
        # Utility
        resume=arg.resume,
        # Hyper-parameter
        numepoches=arg.numepoches,
        lr=arg.lr,
        batchsize=arg.batchsize,
    )
    model.run()


class MY_CNN():
    def __init__(self, lr,numepoches, batchsize, resume, train_loader, test_loader):
        self.numepoches = numepoches
        self.lr = lr
        self.batch_size = batchsize
        self.resume = resume
        self.trainloader = train_loader
        self.testloader = test_loader
        self.best_prec1 = 0

    def build_model(self):
        # our model
        self.net = ResNet18()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        # Define loss (Cross-Entropy)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        self.net.cuda()
        # SGD with momentum
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        #print(torch.cuda.is_available())


    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                      .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))

    def run(self):
        self.build_model()
        # net.train(mode=True)
        self.resume_and_evaluate()
        self.train()
        self.test()

    def train(self):

        for epoch in range(self.numepoches):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data

                # warp them in Variable
                # inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()  # 清空上一步的梯度

                # forward
                outputs = self.net(inputs)
                # loss
                loss = self.criterion(outputs, labels)
                # backward
                loss.backward()
                # update weights
                self.optimizer.step()

                # print statistics
                running_loss = running_loss + loss.data.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


            # prec1, val_loss = self.validate_1epoch()  # 验证
            # print(prec1)
            is_best = running_loss > self.best_prec1
            # lr_scheduler
            # self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = running_loss


            self.save_checkpoint(self.net.state_dict(),is_best, 'record/checkpoint.pth.tar', )
        print("Finished Training")

    def save_checkpoint(self,state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'record/model_best.pth.tar')



    def test(self):
        print("Beginning Testing")
        correct = 0
        total = 0
        for data in self.testloader:
            if torch.cuda.is_available():
                inputs, labels = data
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            #images, labels = data
            outputs = self.net(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))



if __name__ ==  '__main__':
    main()