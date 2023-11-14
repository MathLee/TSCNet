import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.TSCNet_models import TSCNet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=70, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
model = TSCNet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = './dataset/train_dataset/ORSSD/train/image/'
gt_root = './dataset/train_dataset/ORSSD/train/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        s2, s3, s4, s2_sig, s3_sig, s4_sig = model(images)
        # bce+iou
        loss2 = CE(s2, gts) + IOU(s2_sig, gts)
        loss3 = CE(s3, gts) + IOU(s3_sig, gts)
        loss4 = CE(s4, gts) + IOU(s4_sig, gts)
        loss = loss2 + loss3 + loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 100 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))

    save_path = 'models/TSCNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'TSCNet.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
