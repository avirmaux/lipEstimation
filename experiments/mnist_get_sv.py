# Compute AlexNet 50 highest singular vectors for every convolutions
import torch

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

from models.mnist_5 import mnist_5

n_sv = 500

def spec_mnist(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=500,
                use_cuda=True)
        self.spectral_norm = s
        self.u = u
        self.v = v

def save_singular(mnist):
    # Save for convolutions
    torch.save(mnist.conv1.spectral_norm, open('mnist_save_5/conv1-singular', 'wb'))
    torch.save(mnist.conv1.u, open('mnist_save_5/conv1-left-singular', 'wb'))
    torch.save(mnist.conv1.v, open('mnist_save_5/conv1-right-singular', 'wb'))

    torch.save(mnist.conv2.spectral_norm, open('mnist_save_5/conv2-singular', 'wb'))
    torch.save(mnist.conv2.u, open('mnist_save_5/conv2-left-singular', 'wb'))
    torch.save(mnist.conv2.v, open('mnist_save_5/conv2-right-singular', 'wb'))

    torch.save(mnist.conv3.spectral_norm, open('mnist_save_5/conv3-singular', 'wb'))
    torch.save(mnist.conv3.u, open('mnist_save_5/conv3-left-singular', 'wb'))
    torch.save(mnist.conv3.v, open('mnist_save_5/conv3-right-singular', 'wb'))

    torch.save(mnist.conv4.spectral_norm, open('mnist_save_5/conv4-singular', 'wb'))
    torch.save(mnist.conv4.u, open('mnist_save_5/conv4-left-singular', 'wb'))
    torch.save(mnist.conv4.v, open('mnist_save_5/conv4-right-singular', 'wb'))

    torch.save(mnist.conv5.spectral_norm, open('mnist_save_5/conv5-singular', 'wb'))
    torch.save(mnist.conv5.u, open('mnist_save_5/conv5-left-singular', 'wb'))
    torch.save(mnist.conv5.v, open('mnist_save_5/conv5-right-singular', 'wb'))


if __name__ == '__main__':
    clf = mnist_5()
    clf = clf.cuda()
    clf = clf.eval()

    for p in clf.parameters():
        p.requires_grad = False

    compute_module_input_sizes(clf, [1, 1, 28, 28])
    execute_through_model(spec_mnist, clf)

    save_singular(clf)
