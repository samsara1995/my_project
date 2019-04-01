
import torch as t
from torch.autograd import Variable as V
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tqdm import *


t.manual_seed(1000)

def  img2vector(filename):
    rows = 32
    cols = 32
    imgVector = np.zeros((1, rows * cols))
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            if ' ' in lineStr[col]:
                #col=col+1
                num=0
                continue
            elif lineStr[col]=='0':
                num=0
            elif (lineStr[col]!='0') and (lineStr[col]!=' '):
                num=num*10+int(lineStr[col])
            imgVector[0, row * 32 + col] = num

    #print(imgVector)
    return imgVector

# 加载数据集
def loadDataSet():
    # # step 1: 读取训练数据集
    print ("---Getting training set...")
    dataSetDir = '/nfs/syzhou/github/project/dataset/MNIST_data/'
    trainingFileList = os.listdir(dataSetDir + 'traintxtimg/')  # 加载测试数据
    numSamples = len(trainingFileList)

    train_x = np.zeros((numSamples, 1024))
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]

        # get train_x
        train_x[i, :] = img2vector(dataSetDir + 'traintxtimg/%s' % filename)

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[1].split('.')[0]) # return 1
        train_y.append(label)

    # # step 2:读取测试数据集
    print ("---Getting testing set...")
    testingFileList = os.listdir(dataSetDir + 'txtimg') # load the testing set
    numSamples = len(testingFileList)
    test_x = np.zeros((numSamples, 1024))
    test_y = []
    for i in range(numSamples):
        filename = testingFileList[i]

        # get train_x
        test_x[i, :] = img2vector(dataSetDir + 'txtimg/%s' % filename)

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[1].split('.')[0]) # return 1
        test_y.append(label)

    return train_x, train_y, test_x, test_y


def get_fake_data(batch_size=20):
    x=t.rand(batch_size,1)*20
    #print(x)
    y=x*2+(1+0.5*t.rand(batch_size,1))*3
    #print(y)
    return x,y



print ("step 1: load data...")
train_x, train_y, test_x, test_y = loadDataSet()
print(len(test_x))
#print(train_x,train_y,test_x,test_y)


w=V(t.rand(1024,10),requires_grad=True)
b=V(t.zeros(1,10),requires_grad=True)
w.cuda()
b.cuda()
lr=0.00001
train_epoch=30

for iii in range(train_epoch):
    sum_output=0
    print('==> Epoch:[{0}/{1}][training stage]'.format(train_epoch, iii))
    for ii in tqdm(range(60000)):
        x=train_x[ii, :]
        y=train_y[ii]
        x1=t.Tensor(1,1024)
        x1[0]=V(t.from_numpy(x))
        a=np.empty((1), np.int)
        a[0]=y
        target=V(t.from_numpy(a))
        x1.cuda()
        target.cuda()

        y_pred=x1.mm(w)#将tensor扩展为参数tensor的大小。
        y_pred=y_pred+b
        input=y_pred
        prediction = F.log_softmax(input, dim=1)

        loss = nn.CrossEntropyLoss()
        loss.cuda()

        output = loss(prediction, target)
        output.backward()
        sum_output=output+sum_output

        w.data.sub_(lr*w.grad.data)
        b.data.sub_(lr*b.grad.data)

        w.grad.data.zero_()
        b.grad.data.zero_()
#        if ii%10000==0 and ii!=0 :
#            print('loss',sum_output/ii)

    result_loss=sum_output/60000
    print('result loss',result_loss,'\n\n')

correct=0
total=0
for ii in range(10000):
    x = test_x[ii, :]
    y = test_y[ii]

    x1 = t.Tensor(1, 1024)
    x1[0] = V(t.from_numpy(x))
    a = np.empty((1), np.int)
    a[0] = y
    target = V(t.from_numpy(a))
    y_pred = x1.mm(w)  # 将tensor扩展为参数tensor的大小。
    # print('y_pred',y_pred.size())
    y_pred = y_pred + b
    _,prediction=t.max(y_pred,1)
    if target==prediction:
        correct=correct+1
    total=total+1
    #if ii!=0 and ii%1000==0:
    #    print(correct/total)
        #print(target,prediction)
    print(correct/total)





#print(w.data.squeeze().item()," ",b.data.squeeze().item())




