#minist数据集是图片格式，这里将其全部转化为txt格式并保存

#/nfs/syzhou/github/project/dataset/MNIST_data/test
import os
import cv2

data_path='/nfs/syzhou/github/project/dataset/MNIST_data/test'
#data_path='/nfs/syzhou/github/project/dataset/MNIST_data/row'

all_data_path = os.listdir(data_path)


all_data_path.sort(key=lambda x: int(x[:-4]))
num = 0
label_add='/nfs/syzhou/github/project/dataset/MNIST_data/label.txt'
#label_add='/nfs/syzhou/github/project/dataset/MNIST_data/trainlabel.txt'
label = open(label_add, 'r')
for files in all_data_path:
    #print(all_data_path)
    #print(data_path)
    #print(files)
    pic_add=data_path+'/'+files
    print(pic_add)
    image=cv2.imread(pic_add)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    yuzhi, c = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)
    c = cv2.resize(c, (32, 32))

    filesname=files.split('.')[0]
    mystr = label.readline()
    #print(mystr)
    #print(filesname)
    mystr=mystr.split('\n')[0]
    #print(mystr)
    filesname=filesname+'_'+mystr
    print(filesname)
    #"1_18.txt"
    fname = '/nfs/syzhou/github/project/dataset/MNIST_data/txtimg/'+filesname+'.txt'
    #fname = '/nfs/syzhou/github/project/dataset/MNIST_data/traintxtimg/' + filesname + '.txt'
    f = open(fname, 'w')
    for i in c:
        for j in i:
            if j!=0:
                j=255
            f.write(str(j) + ' ')  # 在每个数字左边填写空格，使之变成3位数
        f.write('\n')
    f.close()