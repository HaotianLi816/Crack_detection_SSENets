import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FLAGS
import math

class BaseBlock_ori(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t=6, downsample=False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """
        super(BaseBlock_ori, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel)

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)

        # for main path:
        c  = t * input_channel
        #c = middle_channel
        # 1x1   point wise conv
        self.t = t
        if self.t != 1:
            self.conv1 = nn.Conv2d(input_channel, c, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=self.stride, padding=1, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        # main path
        x = inputs
        if self.t != 1:
            x = F.relu6(self.bn1(self.conv1(inputs)), inplace=True)
            x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu6(self.bn2(self.conv2(inputs)), inplace=True)
            x = self.bn3(self.conv3(x))
        x = x + inputs if self.shortcut else x
        # shortcut path
        return x


class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, middle_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        #c  = t * input_channel
        c = middle_channel
        # 1x1   point wise conv
        self.t = t
        if self.t!= 1:
            self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        # main path
        x = inputs
        if self.t != 1:
            x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
            x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu6(self.bn2(self.conv2(inputs)), inplace=True)
            x = self.bn3(self.conv3(x))
        x = x + inputs if self.shortcut else x
        # shortcut path
        return x


class BaseBlock_MobileNetV1(nn.Module):
    alpha = 1

    def __init__(self, inp, oup, stride = 1, bias = False):
        super().__init__()
        """
            subsample:    whether downsample
        """

        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=bias)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=bias)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU(inplace=True)

        # Shortcut downsampling
        self.weights_init()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




class BaseBlock_ResNet56(nn.Module):
    alpha = 1

    def __init__(self, inp, mip, oup, subsample = False):
        super().__init__()
        """
            subsample:    whether downsample
        """
        s = 0.5 if subsample else 1.0

        self.conv1 = nn.Conv2d(inp, mip, 3, int(1 / s), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(mip, oup, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        if getattr(FLAGS, 'weights_init_Xavier', True):
            self.weights_init()
        else:
            self.weights_init_KaiM()


    def shortcut(self, z, x):
        if x.shape != z.shape:
            # d = self.downsample(x)
            # p = torch.mul(d, 0)
            return z  # + torch.cat((d,p), dim=1)
        else:
            return z + x

    def forward(self, x, shortcuts=False):

        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.conv2(z)
        z = self.bn2(z)

        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)

        return z

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)

    print(x.shape)
    BaseBlock.alpha = 0.5
    b = BaseBlock(6, 5, downsample = True)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())