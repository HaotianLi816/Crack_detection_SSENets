from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
# from ssenets import MyResNet8, cnn8

import time
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
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
        return img, label

def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

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


            # Iterate over data.迭代数据
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels = data

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
                if count_batch%10 == 0:
                    # a = 0.0
                    batch_loss = running_loss / (batch_size*count_batch)
                    running_corrects =  running_corrects.item()
                    batch_acc = running_corrects / (batch_size*count_batch)
                    # a=running_corrects / (batch_size*count_batch)
                    # print(running_corrects)
                    # print(len(preds))
                    # print(batch_size*count_batch)
                    # print(a)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    # print(batch_acc)
                    begin_time = time.time()
            # running_corrects=running_corrects.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # running_corrects = running_corrects.item()
            print(running_corrects)
            print(dataset_sizes[phase])
            print(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model
            if phase == 'train'and (epoch%10==0):
                if not os.path.exists('output_resnet18'):
                    os.makedirs('output_resnet18')
                torch.save(model, 'output_resnet18/dasppp_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            print('Best Acc: {:.4f}'.format(best_acc))
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

    batch_size = 16
    num_class = 2

    image_datasets = {x: customData(img_path='./picture',
                                    txt_path='./' + 'txtjiu' + '/' + x + '.txt',
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # get model and replace the original fc layer with your fc layer


    # model_ft = models.resnet50(pretrained=False)

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


    # model_ft = torch.load("model.pkl")

    # if isinstance(model_ft,torch.nn.DataParallel):
    #     model_ft = model_ft.module

    # resume_path = "output/best_resnet45.pkl"
    #
    # if resume_path != None:
    #     print('loading checkpoint {}'.format(resume_path))
    #     checkpoint = torch.load(resume_path)
    #     #print(checkpoint)
    #     model_ft.load_state_dict(checkpoint['model_ft'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(checkpoint['epoch'])
    # resnet50 = models.resnet50(pretrained=True)
    # print(resnet50)
    ####
    # resume_path = "output/best_resnet45.pkl"
    # #
    # if resume_path != None:
    #     print("=>loading model '{}'".format(resume_path))
    #     checkpoint = torch.load(resume_path)
    #     print(checkpoint)
    #     epoch = checkpoint['epoch']
    #
    #     print(epoch)
    #     model_ft.load_state_dict(checkpoint['state_dict'])
    #     print("=>load model success, start epoch: '{}'".format(epoch))

    # 原文：https: // blog.csdn.net / qq_21578849 / article / details / 86573043


    model_ft = models.resnet50(pretrained=False)
    # pretrained_dict = resnet50.state_dict()
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
    ####


    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)

    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0000001, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)原来是0.001

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.3)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])
    # CUDA_VISIBLE_DEVICES = 0,1
    # model_ft = torch.nn.DataParallel(model_ft,device_ids=[0])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=100,
                           use_gpu=use_gpu)

    # save best model
    # torch.save({
    #     'epoch': num_epochs,
    #     'state_dict': model_ft.state_dict(),
    # }, "output/best_resnet46.pkl")

    torch.save(model_ft,"output_resnet18/modeldasppp_300.pkl")


    ####设置断点


    # 原文：https: // blog.csdn.net / weixin_43122521 / article / details / 88896996
