# Attation code
#By  怪兽
import torch
from torch import nn
from torch.nn.parameter import Parameter
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)  #b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)#b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)#b,2,h,w
        x = self.conv1(x) #b,1,h,w
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #in 8,32,300,300
        #out 8,32,1,1
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class CBAM(nn.Module):
    def __init__(self , in_planes , ratio = 16 , kernel_size = 7):
        super(CBAM , self).__init__()
        self.ca = ChannelAttention(in_planes , ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self , x):
        #in 8,32,300,300
        #out 8,1,300,300
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        #in 8,32,300,300
        #out 8,32,300,300
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        # return x * y

