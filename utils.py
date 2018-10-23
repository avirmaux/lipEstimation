import numpy as np
import ot

from torch.autograd import Variable

import torch
import torchvision.utils as vutils

# Global parameters
use_cuda = torch.cuda.is_available()

def save_model(model, tag=''):
    torch.save(model.state_dict(), '{}.pth.tar'.format(where, tag))


def load_model(model, where):
    """ If the model has been trained on a GPU, and you want to load the
    weights for a CPU, you can do:

    model.load_state_dict(torch.load('model.pth.tar', map_location=lambda storage, loc: storage))
    """
    try:
        model.load_state_dict(torch.load(where))
    except:
        model.load_state_dict(torch.load(where, map_location=lambda storage, loc: storage))


def clip(model, clip):
    if clip is None:
        return
    for p in model.parameters():
        p.data.clamp_(-clip, clip)  # clamp_ is inplace
    return


def clip_gradient(model, clip):
    if clip is None:
        return
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp_(-clip, clip)
    return


def get_sv_linear_model(model):
    singular_values = []
    for p in model.modules():
        if str(type(p)).find('Linear') != -1:
            _, s, _ = torch.svd(p.weight)
            singular_values.append(s.data.numpy())
    return singular_values

# def get_inner_product_hg_sv(model):
#     inner_products = []
#     for p in model.modules():
#         if str(type(p)).find('Linear') != -1
