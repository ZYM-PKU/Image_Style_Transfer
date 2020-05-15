import torch
import torch.nn as nn
from function import AdaIN
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#训练设备


##############定义前传网络#############
class FPnet(nn.Module):
    def __init__(self,decoder):
        super(FPnet, self).__init__()
        self.net = models.vgg19(pretrained=True).features.eval()#使用预训练完成的vgg16
        self.mseloss = nn.MSELoss()
        self.decode=decoder
        for param in self.net.parameters():
            param.requires_grad = False#对encoder不进行梯度下降

    def encode(self, x,layer=0):#基于vgg19编码提取特定层的特征图
        '''分别提取relu1-1,relu2-1,relu3-1,relu4-1的特征，默认提取relu4-1'''
        layers=[21,2,7,12,21]
        for i in range(layers[layer]):
            x = self.net[i](x)
        return x
    def content_loss(self,feature1,feature2):
        '''计算内容损失'''
        dis=self.mseloss(feature1,feature2)
        return dis
    def style_loss(self,img1,img2):
        '''计算样式损失'''
        #提取四个relu层输出
        features1=list(self.encode(img1,layer=l)for l in range(1,5))
        features2=list(self.encode(img2,layer=l)for l in range(1,5))
        #求取均值与标准差
        mean=torch.Tensor(list(self.mseloss(features1[i].mean(),features2[i].mean())  for i in range(4))).sum()
        std=torch.Tensor(list(self.mseloss(features1[i].std(),features2[i].std())  for i in range(4))).sum()
        dis=mean+std
        return dis


    def forward(self,content,style,alpha=1.0,lamda=10.0,require_loss=True):
        '''一次前传计算损失'''
        content=content.to(device)
        style=style.to(device)
        style_features=self.encode(style)
        content_features=self.encode(content)

        t=AdaIN(content_features,style_features)
        t=alpha*t+(1-alpha)*content_features

        output=self.decode(t)
        if not require_loss:return output

        out_features=self.encode(output)
        return self.content_loss(out_features,t)+self.style_loss(output,style)*lamda,output


############定义镜像解码器#############
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        #layer1
        self.pre11=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1)
        self.relu11=nn.ReLU(inplace=True)
        self.up1=nn.Upsample(scale_factor=2, mode='nearest')
        #layer2
        self.pre21=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu21=nn.ReLU(inplace=True)
        self.pre22=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu22=nn.ReLU(inplace=True)
        self.pre23=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu23=nn.ReLU(inplace=True)
        self.pre24=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24=nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1)
        self.relu24=nn.ReLU(inplace=True)

        self.up2=nn.Upsample(scale_factor=2, mode='nearest')
        #layer3
        self.pre31=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv31=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1)
        self.relu31=nn.ReLU(inplace=True)
        self.pre32=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv32=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1)
        self.relu32=nn.ReLU(inplace=True)
        self.up3=nn.Upsample(scale_factor=2, mode='nearest')
        #layer4
        self.pre41=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv41=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.relu41=nn.ReLU(inplace=True)
        self.pre42=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv42=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1)
        self.relu42=nn.ReLU(inplace=True)
        self.softmax=nn.Softmax2d()

    def forward(self, x):
        x=self.pre11(x)
        x=self.relu11(self.conv11(x))
        x=self.up1(x)
        x=self.pre21(x)
        x=self.relu21(self.conv21(x))
        x=self.pre22(x)
        x=self.relu22(self.conv22(x))
        x=self.pre23(x)
        x=self.relu23(self.conv23(x))
        x=self.pre24(x)
        x=self.relu24(self.conv24(x))
        x=self.up2(x)
        x=self.pre31(x)
        x=self.relu31(self.conv31(x))
        x=self.pre32(x)
        x=self.relu32(self.conv32(x))
        x=self.up3(x)
        x=self.pre41(x)
        x=self.relu41(self.conv41(x))
        x=self.pre42(x)
        x=self.conv42(x)
        return x
 