#https://blog.csdn.net/u014380165/article/details/78634829
#https://https://github.com/miraclewkf/ImageClassification-PyTorch/blob/master~\\/level2/train\_customData.py
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
from aspp import ASPP, ASPP_Bottleneck
# from resnet import RESNET,RESNET_Bottleneck

import torch.nn.functional as F
import time
import os
from torch.utils.data import Dataset
from torchnet import meter
from sklearn.metrics import confusion_matrix


from PIL import Image

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.num_classes = 2
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.conv_1x1_1 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(32)
        self.conv_3x3_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(32)
        self.conv_3x3_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(32)
        self.conv_3x3_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(32)
        self.conv_1x1_3 = nn.Conv2d(160, 32, kernel_size=1)  # (160 = 5*32)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(32)
        self.conv_1x1_4 = nn.Conv2d(32, 64, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # print(x.shape)
        # feature_map_h = x.size()[2]  # (== h/16)
        # feature_map_w = x.size()[3]  # (== w/16)
        # out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
        # out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
        # out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))  # (shape: (batch_size, 256, h/16, w/16))
        # out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))  # (shape: (batch_size, 256, h/16, w/16))
        # # 把输出改为256
        # out_img = self.avg_pool(x)  # (shape: (batch_size, 512, 1, 1))
        # out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        # out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w),
        #                      mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))
        # out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
        #                 1)  # (shape: (batch_size, 1280, h/16, w/16))
        # out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (shape: (batch_size, 256, h/16, w/16))
        # out = self.conv_1x1_4(out)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
# txt文件中每行都是图像路径，tab键，标签
# 定义数据读取接口
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

# load the exist model
model = torch.load("output_aspp/modeldasppp_300.pkl")
since = time.time()

if __name__ == '__main__':
#  Transform中将每张图像都封装成Tensor
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    use_gpu = torch.cuda.is_available()
    begin_time = time.time()
    batch_size = 32
    num_class = 2

# 调用数据读取接口
    image_datasets = {x: customData(img_path='./picture',
                                    txt_path=('./txtjiu/' + x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['val']}
    print(len(image_datasets))

# 将这个batch的图像数据和标签都分别封装成Tensor
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['val']}

#    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

# define cost function
criterion = nn.CrossEntropyLoss()
# criterion = nn.MultiLabelSoftMarginLoss()

#model eval
model.eval()
eval_loss = 0.0
eval_acc = 0.0
a=0
for data in dataloders['val']:
    img,label = data
#    img = img.view(img.size(0), -1)
#    img = Variable(img.cuda())
    img = Variable(img)
    # img = img.type(torch.FloatTensor)
    label = Variable(label.cuda())
    #label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    # eval_loss += loss.data[0] * label.size(0)
    eval_loss += loss.item() * label.size(0)
    _,pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    # eval_acc = eval_acc.float()
    # eval_acc += num_correct.data[0]
    eval_acc += num_correct.data
    eval_acc = eval_acc.item()
    a +=  len(pred)
    confusion_matrix = meter.ConfusionMeter(2)

    confusion_matrix.add(label.data, pred)


    end = time.time()
    time_elapsed = time.time() - since


    # print(data)
    # print(pred)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / a,eval_acc / a))
    print(confusion_matrix.value())
    # print( num_correct)
    # # print( _,pred )
    # print(len(pred))
    # print(eval_loss / (len(pred)))
    #print(eval_acc / (len(pred)))
    print(eval_acc)
    print(a)
    print(time_elapsed)
    # print(eval_loss)
    # print(out)