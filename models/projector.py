import torch
import torch.nn as nn

from torchvision import models

from models.cifar_class import DeepNet

from models.alexnet_truncated import alexnet_truncated


def projector(mode, *args):
    if mode == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.eval()
        return nn.Sequential(nn.Upsample(size=(224, 224), mode='bilinear'),
                             model)
    if mode == 'alextrunc':
        model = alexnet_truncated(pretrained=True)
        model.eval()
        return nn.Sequential(nn.Upsample(size=(224, 224), mode='bilinear'),
                             model)
    if mode == 'deepnet':
        model = DeepNet()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('models/cifar-model.pth.tar'))
        model.eval()
        return model
