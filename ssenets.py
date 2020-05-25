import torchvision.models as models

import torch

import torch.nn as nn

import math

import torch.utils.model_zoo as model_zoo

import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Se_module_diff(nn.Module):
    def __init__(self, inp, oup, Avg_size = 1, se_ratio = 1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((Avg_size, Avg_size))
        num_squeezed_channels = max(1,int(inp / se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=inp, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        self.Avg_size = Avg_size
        self.reset_parameters()

    #x and z are different conv layer and z pass through more convs
    def reset_parameters(self):
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


    def forward(self, x, z):
        SIZE = z.size()
        y = self.avg(x)
        y = self._se_reduce(y)
        y = y * torch.sigmoid(y)
        y = self._se_expand(y)
        if self.Avg_size != 1:
            y = F.upsample_bilinear(y, size=[SIZE[2], SIZE[3]])
        z = torch.sigmoid(y) * z
        return z

class MyResNet8(nn.Module):

    def __init__(self, num_classes=1000):
            super(MyResNet8, self).__init__()
            self.inplanes = 64
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            #                        bias=False)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # _make_layer方法是用来构建ResNet网络中的4个blocks。_make_layer方法的第一个输入block是Bottleneck或BasicBlock类，
            # 第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构.

            # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)

            ######

            in_channels = 256
            out_channels = 4
            self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

            self.conv_3x3_1 = nn.Conv2d(in_channels, in_channels * out_channels, kernel_size=3, stride=1, padding=3,
                                        dilation=3, groups=in_channels)
            self.bn_conv_3x3_1_1 = nn.BatchNorm2d(in_channels * out_channels)
            # self.conv_3x3_1_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
            self.conv_3x3_1_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=1)
            self.bn_conv_3x3_1_2 = nn.BatchNorm2d(out_channels)

            self.conv_3x3_2 = nn.Conv2d(in_channels, in_channels * out_channels, kernel_size=3, stride=1, padding=7,
                                        dilation=7, groups=in_channels)
            self.bn_conv_3x3_2_1 = nn.BatchNorm2d(in_channels * out_channels)
            # self.conv_3x3_2_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
            self.conv_3x3_2_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=1)
            self.bn_conv_3x3_2_2 = nn.BatchNorm2d(out_channels)

            self.conv_3x3_3 = nn.Conv2d(in_channels, in_channels * out_channels, kernel_size=3, stride=1, padding=11,
                                        dilation=11, groups=in_channels)
            self.bn_conv_3x3_3_1 = nn.BatchNorm2d(in_channels * out_channels)
            # self.conv_3x3_3_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
            self.conv_3x3_3_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=1)
            self.bn_conv_3x3_3_2 = nn.BatchNorm2d(out_channels)

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
            self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)  # (160 = 5*32)
            self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
            self.conv_1x1_4 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
            # self.bn_conv_1x1_4 = nn.BatchNorm2d(in_channels)

#########################

            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=2, padding=2, bias=False)
            self.bn4 = nn.BatchNorm2d(512)


            self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2,dilation=2, padding=2, bias=False)
            self.bn5 = nn.BatchNorm2d(1024)



            self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, dilation=2,padding=2, bias=False)
            self.bn6 = nn.BatchNorm2d(2048)

            self.se_module_diff = Se_module_diff(inp=256, oup=2048)

            self.avgpool = nn.AvgPool2d(4, stride=1)
            # self.avgpool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(2048, num_classes)





    def forward(self, x):

                               #输入[16, 3, 224, 224]
        x1 = self.conv1(x)
                               #输出[16, 64, 112, 112]
        x1 = self.bn1(x1)        #输出[16, 64, 112, 112])
        x1 = self.relu(x1)       #输出[16, 64, 112, 112])
        x1 = self.maxpool(x1)    #输出[16, 64, 56, 56]


        x2= self.conv2(x1)      #[16, 128, 56, 56]

        x2 = self.bn2(x2)  # 输出
        x2 = self.relu(x2)  # 输出
        x2 = self.maxpool(x2)    #[16, 128, 28, 28]



        x3 = self.conv3(x2)  # [16, 256, 28, 28]
        x3 = self.bn3(x3)  # 输出
        x3 = self.relu(x3)  # 输出
        x3 = self.maxpool(x3)   #[16, 256, 14, 14]

        # print(x.shape)
    ###########
        # feature_map_h = x3.size()[2]  # (== h/16)
        # feature_map_w = x3.size()[3]  # (== w/16)
        # out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x3)))  # 111(shape: ([16, 256, 14, 14])
        # out_3x3_1 = F.relu(self.bn_conv_3x3_1_2(
        # self.conv_3x3_1_point(self.bn_conv_3x3_1_1(self.conv_3x3_1(x3)))))  # (shape: ([16, 4, 14, 14])
        # out_3x3_2 = F.relu(self.bn_conv_3x3_2_2(
        # self.conv_3x3_2_point(self.bn_conv_3x3_2_1(self.conv_3x3_2(x3)))))  # (shape: ([16, 4, 14, 14])
        # out_3x3_3 = F.relu(self.bn_conv_3x3_3_2(
        # self.conv_3x3_3_point(self.bn_conv_3x3_3_1(self.conv_3x3_3(x3)))))  # (shape: [16, 4, 14, 14])
        #
        # # 把输出改为256
        # out_img = self.avg_pool(x3)  # (shape: ([16, 256, 1, 1])
        # out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: [16, 4, 1, 1])
        # out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w),mode="bilinear")  # (shape: ([16, 4, 14, 14])
        #
        # x = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],1)  # (shape: ([16, 20, 14, 14])
        # x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))  # (shape: [16, 4, 14, 14])
        # x4 = self.conv_1x1_4(x)  #[16, 256, 14, 14]


    ###########


        x4 = self.conv4(x3)
        x4 = self.bn4(x4)  # 输出
        x4 = self.relu(x4)  # 输出
        # x = self.maxpool(x)  # [16, 512, 7, 7]



        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)     #[16, 1024, 8, 8]


        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = self.relu(x6)   #[16, 2048, 4, 4]
        x6 = self.se_module_diff(x3, x6)
        # print(x.shape)
        # x = self.AdaptiveAvgPool2d(1)
        x = self.avgpool(x6)
        # print(x.shape)     #[16, 1024, 1, 1]
        x = x.view(x.size(0), -1)   #输出[16, 160000] 将第二次卷积的输出拉伸为一行[16,64*50*50]
        # print(x.shape)
        x = self.fc(x)

        return x


# resnet50 = models.resnet50(pretrained=True)
cnn8 = MyResNet8()

print(cnn8)
torch.save(cnn8, 'model.pkl')

