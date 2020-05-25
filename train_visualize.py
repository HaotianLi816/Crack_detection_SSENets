from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
from dilatedaspp_se import MyResNet8, cnn8
# from resnet4 import ResNet, Bottleneck,cnn
# from aspp import ASPP, ASPP_Bottleneck
import time
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import visdom as vis


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         # self.num_classes = 2
#         # 加了这句
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#         #                        bias=False)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#
#         in_channels = 256
#         out_channels = 256
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # _make_layer方法是用来构建ResNet网络中的4个blocks。_make_layer方法的第一个输入block是Bottleneck或BasicBlock类，
#         # 第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构.
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#
#         # self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         # self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
#         # self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
#         # self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
#         # self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4)
#         # self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
#         # self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=8, dilation=8)
#         # self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         # self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
#         # self.conv_1x1_3 = nn.Conv2d(out_channels*5, out_channels, kernel_size=1)  # (160 = 5*32)
#         # self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
#         # self.conv_1x1_4 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
#         #
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         # self.avgpool = nn.AvgPool2d(7, stride=1)
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
# # 拉拉
#         self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=in_channels)
#         self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels*in_channels)
#         self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels)
#         self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels*in_channels)
#         self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4, groups=in_channels)
#         self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels*in_channels)
#         self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=8, dilation=8, groups=in_channels)
#         self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels*in_channels)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels*in_channels)
#         self.conv_1x1_3 = nn.Conv2d(in_channels * out_channels * 5, in_channels * out_channels, kernel_size=1)  # (160 = 5*32)
#         self.bn_conv_1x1_3 = nn.BatchNorm2d(in_channels * out_channels)
#         self.conv_1x1_4 = nn.Conv2d(in_channels * out_channels, in_channels, kernel_size=1)
#
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         # 将每个blocks的第一个residual结构保存在layers列表中。
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#             # 将每个blocks的剩下residual
#             # 结构保存在layers列表中，这样就完成了一个blocks的构造。
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#
#         # img = np.array(img)
#         #
#         #print(x.shape)
#
#
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
# # 拉拉
#         feature_map_h = x.size()[2]  # (== h/16)
#         feature_map_w = x.size()[3]  # (== w/16)
#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))  # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))  # (shape: (batch_size, 256, h/16, w/16))
#         # 把输出改为256
#         out_img = self.avg_pool(x)  # (shape: (batch_size, 512, 1, 1))
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
#         out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w),
#                              mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))
#         x = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
#                         1)  # (shape: (batch_size, 1280, h/16, w/16))
#         x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))  # (shape: (batch_size, 256, h/16, w/16))
#         x = self.conv_1x1_4(x)
#
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
# # resnet50 = models.resnet50(pretrained=True)
# cnn = ResNet(Bottleneck, [3, 4, 6, 3])
# # 读取参数
# # pretrained_dict = resnet50.state_dict()
# # model_dict = cnn.state_dict()
# # # 将pretrained_dict里不属于model_dict的键剔除掉
# # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # # 更新现有的model_dict
# # model_dict.update(pretrained_dict)
# # # 加载我们真正需要的state_dict
# # cnn.load_state_dict(model_dict)
# # print(resnet50)
# # print(cnn)
# # torch.save(cnn, 'model.pkl')
# # torch.save(cnn.state_dict(),'model.pkl')
# # torch.save(cnn,'model2.pkl')
# torch.save(cnn.state_dict(),'model2.pth')
#

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
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
        name=img_name.split("\\")[1]
        label = self.img_label[item]
        img = self.loader(img_name)
        # img = np.array(img)
        #
        # print(img.shape)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label, name

def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu, vis = None):

    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    cnt = 0

    for epoch in range(num_epochs):
        begin_time = time.time()
        count_batch = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase 训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            lines = []
            # Iterate over data.迭代数据
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels, name = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                '''
                if phase == 'val'and epoch==num_epochs-1:
                    lines.append(preds)
                '''

                loss = criterion(outputs, labels)
                # print(_, preds)
                # print(loss)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                # print result every 10 batch
                if count_batch%1 == 0:
                    # a = 0.0
                    batch_loss = running_loss / (batch_size*count_batch)
                    running_corrects =  running_corrects.item()
                    batch_acc = running_corrects / (batch_size*count_batch)
                    # a=running_corrects / (batch_size*count_batch)
                    # print(running_corrects)
                    # print(len(preds))
                    # print(batch_size*count_batch)
                    # print(a)
                    #print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                    #      format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    # print(batch_acc)
                    begin_time = time.time()
                    if vis != None and phase == 'train':
                        vis.line("train/batch_loss", batch_loss, cnt)
                        vis.line("train/batch_acc", batch_acc, cnt)
                        cnt += 1

            # running_corrects=running_corrects.item()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if (vis != None and phase == 'train'):
                vis.line("train/epoch_loss", epoch_loss, epoch)
                vis.line("train/epoch_acc", epoch_acc, epoch)
            elif (vis != None and phase == 'val'):
                vis.line("val/epoch_loss", epoch_loss, epoch)
                vis.line("val/epoch_acc", epoch_acc, epoch)
            # running_corrects = running_corrects.item()
            print(running_corrects)
            print(dataset_sizes[phase])
            #print(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model
            if phase == 'train':
                if not os.path.exists('output_aspp_pre_se_S1_E2_3'):
                    os.makedirs('output_aspp_pre_se_S1_E2_3')
                torch.save(model, 'output_aspp_pre_se_S1_E2_3/dasppp_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    use_gpu = torch.cuda.is_available()

    batch_size = 32
    num_class = 2

    image_datasets = {x: customData(img_path='./picture',
                                    txt_path='./' + 'txtjiu' + '/' + x + '.txt',
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=16) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # get model and replace the original fc layer with your fc layer
    # model_ft = models.resnet50(pretrained=True)
    # 预训练
    # resnet50 = models.resnet50(pretrained=True)
    # pretrained_dict = resnet50.state_dict()
    # cnn.load_state_dict(torch.load( 'model2.pth'))
    # model_dict = model.state_dict()
    #
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)


    # model_ft = torch.load("model5.pkl")
    # resnet50 = models.resnet50(pretrained=True)
    # print(resnet50)
    ####
    # model_ft = models.resnet50(pretrained=True)
    # resnet50 = models.resnet50(pretrained=True)
    # pretrained_dict = resnet50.state_dict()
    #
    # # torch.save(resnet50.state_dict(), "resnet.pth")
    # # 加载resnet，模型存放在my_resnet.pth
    # # resnet50.load_state_dict(torch.load("resnet.pth"))
    # """加载torchvision中的预训练模型和参数后通过state_dict()方法提取参数
    # 也可以直接从官方model_zoo下载：
    # pretrained_dict = model_zoo.load_url(model_urls['resnet152'])"""
    # # model_dict = model.state_dict()
    # model_dict = cnn.state_dict()
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的state_dict
    # # cnn.load_state_dict(model_dict)
    # cnn.load_state_dict(model_dict)
    # torch.save(cnn, "my_resnet2.pth")
    # model_ft = cnn
    # ###
    #
    # vis = vis.Visualizer()
    #
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)

    model_ft = torch.load("model.pkl")
    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.3)


    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])
    # model_ft = torch.nn.DataParallel(model_ft)

    # train model
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=300,
                           use_gpu=use_gpu,
                           vis = vis)

    # save best model
    torch.save(model_ft,"output_aspp_pre_se_S1_E2_3/modeldasppp_300.pkl")
