from torch import nn


__all__ = ['GradNet', 'gradnet']



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=-1, dilation=1, groups=1):
        if padding < 0:
            padding = ((kernel_size - 1) // 2) * dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class OuterProj(nn.Module):
    '''
    '''
    def __init__(self, ich, pch, och, kernel_size=3, stride=1, padding=-1, dilation=1):
        #
        super(OuterProj, self).__init__()
        #
        if padding < 0:
            padding = ((kernel_size - 1) // 2) * dilation
        self.conv_proj = nn.Sequential(
            nn.Conv2d(ich, pch, kernel_size=1, bias=False),
            nn.BatchNorm2d(pch)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(pch*pch, pch*pch, kernel_size, stride, padding, dilation, groups=pch*pch, bias=False),
            nn.BatchNorm2d(pch*pch),
            ConvBNReLU(pch*pch, och, 1)
        )

    def forward(self, x):
        out = self.conv_proj(x)

        # outer product
        pch = out.size(1)
        lhs = out.repeat((1, pch, 1, 1))
        rhs = out.repeat_interleave(pch, dim=1)
        out = lhs * rhs

        out = self.conv_out(out)
        return out


class GradNet(nn.Module):
    def __init__(self, out_channels):
        """
        """
        super(GradNet, self).__init__()

        self.conv_in = ConvBNReLU(3, 16, kernel_size=3, stride=2, padding=0)

        blocks = [
            OuterProj(16, 8, 16, 3, stride=1),
            OuterProj(16, 8, 32, 3, stride=2, padding=0),
            OuterProj(32, 16, 32, 3, stride=1),
            OuterProj(32, 16, 64, 3, stride=2, padding=0),
            OuterProj(64, 32, 64, 3, stride=1, dilation=2),
            OuterProj(64, 32, 64, 3, stride=1, dilation=2)
        ]
        self.features = nn.ModuleList(blocks)

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv_in(x)

        for feat in self.features:
            o = feat(out)
            if out.size(1) == o.size(1):
                out = out + o
            else:
                out = o

        out = self.conv_out(out)
        return out


def gradnet(**kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = GradNet(**kwargs)
    return model
