# Compute AlexNet 50 highest singular vectors for every convolutions
import torch
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

n_sv = 200

def spec_alex(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=500, use_cuda=True)
        self.spectral_norm = s
        self.u = u
        self.v = v

        # print(self.__class__.__name__, s)
    if is_batch_norm(self):
        # One could have also used generic_power_method
        s = lipschitz_bn(self)
        self.spectral_norm = s
        # print(self.__class__.__name__, s)

def save_singular(alex):
    # Save for convolutions
    feats = alex.features
    for i in range(len(feats)):
        if hasattr(feats[i], 'spectral_norm'):
            torch.save(feats[i].spectral_norm, open('alex_save/feat-singular-{}-{}'.format(feats[i].__class__.__name__, i), 'wb'))
        if hasattr(feats[i], 'u'):
            torch.save(feats[i].u, open('alex_save/feat-left-sing-{}-{}'.format(feats[i].__class__.__name__, i), 'wb'))
            torch.save(feats[i].v, open('alex_save/feat-right-sing-{}-{}'.format(feats[i].__class__.__name__, i), 'wb'))
    # Save for classification
    clf = alex.classifier
    for i in range(len(clf)):
        if hasattr(clf[i], 'spectral_norm'):
            torch.save(clf[i].spectral_norm, open('alex_save/feat-singular-{}-{}'.format(clf[i].__class__.__name__, i), 'wb'))
        if hasattr(clf[i], 'u'):
            torch.save(clf[i].u, open('alex_save/feat-left-sing-{}-{}'.format(clf[i].__class__.__name__, i), 'wb'))
            torch.save(clf[i].v, open('alex_save/feat-right-sing-{}-{}'.format(clf[i].__class__.__name__, i), 'wb'))


if __name__ == '__main__':
    alex = torchvision.models.alexnet(pretrained=True)
    alex = alex.cuda()

    for p in alex.parameters():
        p.requires_grad = False

    compute_module_input_sizes(alex, [1, 3, 224, 224])
    execute_through_model(spec_alex, alex)

    save_singular(alex)
