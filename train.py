'''

Arbitrary style transfer in Real-time with Pytorch.

Run the script with:
```
python train.py -c path_to_your_content_dir \
    -s path_to_your_style_dir 
```
e.g.:
```
python train.py -c content -s style

```
Optional parameters:
```
--save_dir, The path to save your model(Default is "model/decoder.pth")
--lr,  Learning rate (Default is 1e-3)
--iter_times, Times of the iteration(max epoch) (Default is 1000)
--batch_size, (Default is 4)
--lamda The weight of the style loss (Default is 10.0)
--alpha content-style trade-off (Default is 1.0)
--n_threads Numworkers (Default is 0 on Windows)
```

It is preferable to run this script on GPU, for speed.

# References
    - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization]-Xun Huang & Serge Belongie

'''

import os
import cv2
import time
import glob
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsummary import summary
from net import FPnet,Decoder
from PIL import Image, ImageFile



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#训练设备



####################options###################
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-c','--content_dir',type=str, required=True,
                    help='Directory path to batchs of content images')
parser.add_argument('-s','--style_dir',type=str, required=True,
                    help='Directory path to batchs of style images')
# Training options
parser.add_argument('--save_dir', default='model/decoder.pth',
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('-i','--iter_times', type=int, default=1000)
parser.add_argument('-b','--batch_size', type=int, default=4)
parser.add_argument('--lamda', type=float, default=10.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=0)


args = parser.parse_args()



#############定义训练数据集#############
class CSDataset(data.Dataset):
    def __init__(self, root, transform):
        super(CSDataset, self).__init__()
        self.root = root
        self.paths = glob.glob(self.root+'/*.jpg')
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'CSDataset'


def Transform():
    '''图像缩放剪切转换'''
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)



def train(decoder,content_loader,style_loader):
    '''训练解码器'''
    print('Start training...')
    tic=time.time()
    net=FPnet(decoder)
    net=net.to(device)
    #使用adam优化器
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    #optimizer = optim.SGD(decoder.parameters(), lr=0.001, momentum=0.9) #优化函数为随机梯度下降
    optimizer.zero_grad()##梯度初始化
    for epoch in range(args.iter_times):
        loss_data=[]
        for i,content in enumerate(content_loader,0):
            for j,style in enumerate(style_loader,0):
                loss,output=net(content,style,lamda=args.lamda,alpha=args.alpha)
                loss_data.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_data=torch.tensor(loss_data, dtype=torch.float)
        print('iters: '+str(epoch)+'  loss:'+str(loss_data.mean().item()))
    torch.save(decoder.state_dict(), args.save_dir)
    toc=time.time()
    print('TRIANING COMPLETED.')
    print('Time cost: {}s.'.format(toc-tic))






def main():

    #定义数据加载器
    transform=Transform()
    content_set = CSDataset(transform=transform, root=args.content_dir)
    style_set = CSDataset(transform=transform, root=args.style_dir)
    content_loader = data.DataLoader( content_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads,pin_memory=True,drop_last=True) 
    style_loader = data.DataLoader( style_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads,pin_memory=True,drop_last=True)

    ########初始化模型########
    decoder=Decoder()
    decoder=decoder.to(device)
    decoder.zero_grad()
    ##########################

    #训练网络
    train(decoder,content_loader,style_loader)


if __name__ == "__main__":
    main()