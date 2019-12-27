import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18 as _resnet18
from torchvision.models import resnet34 as _resnet34
from torchvision.models import resnet50 as _resnet50

__all__ = ['ResnetCGD', 'resnet18', 'resnet34', 'resnet50']


class NormLayer(nn.Module):
    def __init__(self, kernel_size, padding=(0, 0), eps=1e-06):
        '''
        '''
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if kernel_size[0] > 1 or kernel_size[1] > 1:
            self.pool = nn.AvgPool2d(kernel_size, 1, padding)
        else:
            self.pool = None
        self.eps = eps
        
    def forward(self, x):
        u_x = torch.mean(x, dim=1, keepdim=True)
        u_x2 = torch.mean(x*x, dim=1, keepdim=True)
        if self.pool is not None:
            u_x = self.pool(u_x)
            u_x2 = self.pool(u_x2)
        v_x = F.relu(u_x2 - (u_x * u_x), inplace=True)
        out = x / torch.sqrt(v_x + self.eps)
        return out
    
    
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
    

class ResnetCGD(nn.Module):
    '''
    '''
    def __init__(self, backbone='resnet18', pretrained=False):
        '''
        '''
        super().__init__()
        net = self._get_backbone(backbone)(pretrained=pretrained)
        self.base_layers = nn.ModuleList(list(net.children())[:-2])
        self.base_names = [l[0] for l in net.named_children()]
        
        self.color_layer = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            NormLayer(3, 1, 1),
            nn.AvgPool2d(2, 2, padding=0),
        )
        
        self.grad_norm_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            NormLayer(3, 1, 1),
            nn.AvgPool2d(2, 2, padding=0),
        )
        
        # self.downsample_layer = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2),
        #     NormLayer(3, 1, 1),
        # )
        
        self.upsample_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            DepthToSpace(2),
            NormLayer(3, 1, 1),
        )
        
        self.mid_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            NormLayer(3, 1, 1),
        )
        
    def _get_backbone(self, name):
        if name == 'resnet18':
            return _resnet18
        elif name == 'resnet34':
            return _resnet34
        elif name == 'resnet50':
            return _resnet50
        else:
            raise ValueError('Not supported backbone')

    def forward(self, x):
        '''
        '''
        # color layer
        color_feat = self.color_layer(x)
        
        # base layers
        base_layers = {}
        for n, layer in zip(self.base_names, self.base_layers):
            x = layer(x)
            base_layers[n] = x
            
        # grad layer
        grad_feat = self.grad_norm_layer(base_layers['layer1'])
        
        # deep feature layer
        # fd = self.downsample_layer(base_layers['layer2'])
        fm = self.mid_layer(base_layers['layer3'])
        fu = self.upsample_layer(base_layers['layer4'])
        # deep_feat = torch.cat([fd, fm, fu], dim=1)

        feat = torch.cat([color_feat, grad_feat, fm, fu], dim=1)
        return feat


def resnet18(**kwargs):
    return ResnetCGD(backbone='resnet18', **kwargs)


def resnet34(**kwargs):
    return ResnetCGD(backbone='resnet34', **kwargs)


def resnet50(**kwargs):
    return ResnetCGD(backbone='resnet50', **kwargs)