import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

import timm
from timm.models.layers import to_2tuple,trunc_normal_

from utils.layers import ConvBlock, init_layer, init_bn


class weak_mxh64_1024(nn.Module):

    def __init__(self,nclass,glplfn,params):
        super(weak_mxh64_1024,self).__init__() 

        self.spec_augmenter = SpecAugmentation(time_drop_width=params.time_drop_width, time_stripes_num=params.time_stripes_num,
                                               freq_drop_width=params.freq_drop_width, freq_stripes_num=params.freq_stripes_num)

        self.globalpool = glplfn
        self.layer1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(32,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer6 = nn.MaxPool2d(2)

        self.layer7 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer9 = nn.MaxPool2d(2)

        self.layer10 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer12 = nn.MaxPool2d(2)

        self.layer13 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer15 = nn.MaxPool2d(2) #

        self.layer16 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        self.layer17 = nn.MaxPool2d(2) # 
        
        self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.BatchNorm2d(1024),nn.ReLU())
        self.layer19 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())

    def forward(self,x):
        if self.training:
            x = self.spec_augmenter(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out1 = self.layer19(out)
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)
        return out 


class Cnn14(nn.Module):
    def __init__(self, params):

        super(Cnn14, self).__init__()

        self.spec_augmenter = SpecAugmentation(time_drop_width=params.time_drop_width, time_stripes_num=params.time_stripes_num,
                                               freq_drop_width=params.freq_drop_width, freq_stripes_num=params.freq_stripes_num)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, params.num_class, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x)) 
        # clipwise_output = self.fc_audioset(x)

        return clipwise_output

