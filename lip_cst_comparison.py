# encoding: utf-8

import os
import argparse

import numpy as np

import scipy.stats as st

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable

from torchvision import datasets, transforms, models

from models.mlp import MLP
# from simplenn import MultiLayerPercetron
# from models.multi_layer_perceptron import MultiLayerPercetron, MultiLayerPercetronNoActivation

from lipschitz_utils import *
import experiments.gp as gp

from models.actors import actor
from models.projector import projector

from lipschitz_approximations import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import clip, clip_gradient, save_images, sample_draws, compute_wasserstein

# Global parameters
use_cuda = torch.cuda.is_available()

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='MNIST | CIFAR')
#parser.add_argument('--root', required=True, help='path to dataset')
#parser.add_argument('--batchSize', type=int, default=128, help='size of input batch')

#opt = parser.parse_args()
#print(opt)

def create_dataset(data_type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # MNIST
    if data_type is 'MNIST':
        return datasets.MNIST(root='data/', download=True, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))

    # CIFAR
    if data_type is 'CIFAR':
        return datasets.CIFAR10(root='data/', download=True, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))

    # RANDOM
    if data_type is 'RANDOM':
        input_size = 10
        return random_dataset([input_size], 1000)


def compute_lipschitz_approximations(model, data):
    print(model)
    # Don't compute gradient for the projector: speedup computations
    for p in model.parameters():
        p.requires_grad = False

    # Compute input sizes for all modules of the model
    input_size = get_input_size(data)
    print(input_size)
    compute_module_input_sizes(model, input_size)

    # Lipschitz lower bound through optimization of the gradient norm
    print('Computing lip_opt...')
    lip_opt = lipschitz_annealing(model, n_iter=1000, temp=1, batch_size=1)

    # Lipschitz lower bound on dataset
    print('Computing lip_data...')
    lip_data = lipschitz_data_lb(model, data, max_iter=1000)

    # Lipschitz upper bound using the product of spectral norm
    print('Computing lip_spec...')
    lip_spec = lipschitz_spectral_ub(model).data[0]

    # Lipschitz upper bound using the product of Frobenius norm
    print('Computing lip_frob...')
    lip_frob = lipschitz_frobenius_ub(model).data[0]

    print('Computing lip_secorder greedy...')
    lip_secorder_greedy = lipschitz_second_order_ub(model, algo='greedy')

    print('Computing lip_secorder BFGS...')
    lip_secorder_bfgs = lipschitz_second_order_ub(model, algo='bfgs')

    # Lipschitz upper bound using the absolute values of the weights
    # WARNING: this computation should be last as so far if changes the model!
    print('Computing lip_abs...')
    lip_abs = lipschitz_absweights_ub(model).data[0]

    print('Lipschitz approximations:\nLB-dataset:\t{:.3f}\n'
          'LB-optim:\t{:.3f}\nUB-frobenius:\t{:.3f}\nUB-spectral:\t{:.3f}\n'
          'UB-absolute:\t{:.3f}\nUB-secorder greedy:\t{:.3f}\n'
          'UB-secorder bfgs:\t{:.3f}'.format(lip_data, lip_opt, lip_frob, lip_spec, lip_abs,
                 lip_secorder_greedy, lip_secorder_bfgs))


def plot_model(model, window_size=1, num_points=100):
    fig = plt.figure()
    ax = Axes3D(fig)
    points = window_size * Variable(torch.randn(num_points, 2))
    fun_val = model(points).data

    ax.scatter(points[:, 0], points[:, 1], fun_val)
    plt.show()


def plot_data(x, y):
    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)

    # z = y[:]
    # x, y = np.meshgrid(x[:, 0], x[:, 1])

    # ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    # ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap=cm.coolwarm)
    ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap=cm.RdYlBu_r)
    plt.show()


def plot_model_data(model, n_points, data=None, data_y=None, coef=1):
    x_train = coef * torch.randn(n_points, 2)
    y = model(Variable(x_train, volatile=True))
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train[:, 0], x_train[:, 1], y.data, c='b')
    if (data is not None) and (data_y is not None):
        ax.scatter(data[:, 0], data[:, 1], data_y, c='g')
    plt.show()


def plot_models(model1, model2, n_points):
    x_train = 200 * torch.randn(n_points, 2)
    y1 = model1(Variable(x_train, volatile=True))
    y2 = model2(Variable(x_train, volatile=True))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train[:, 0], x_train[:, 1], y1.data, c='b')

    ax.scatter(x_train[:, 0], x_train[:, 1], y2.data, c='g')
    plt.show()


def random_matrix_fixed_spectrum(shape, epsilon, K=5):
    if p.shape[0] > 1:
        umat = st.special_ortho_group.rvs(shape[0])
    else:
        umat = np.ones((1, 1))
    if p.shape[1] > 1:
        vmat = st.special_ortho_group.rvs(shape[1])
    else:
        vmat = np.ones((1, 1))

    sigma = epsilon * np.eye(*shape)
    for i in range(min(K, sigma.shape[0], sigma.shape[1])):
        sigma[i, i] = 1
    return torch.Tensor(umat @ sigma @ vmat.transpose())


if __name__ == "__main__":

    print(__doc__)

    vals = range(1, 11)
    plt.semilogy(vals, np.ones(len(vals)), '--', label='AutoLip')
    plt.semilogy(vals, [1/math.pi**(x-1) for x in vals], '--', label='theoretical limit')
    for epsilon in [0.01]:#[0.5, 0.1, 0.01]:
        res = np.ones(len(vals))
        for i in range(len(vals)):
            input_size = 2
            output_size = 1
            layer_size = 100
            n_layers = vals[i] #5
            layers = [layer_size] * n_layers
            model = MLP(input_size, output_size, layers)
            #print(model)

            for p in model.parameters():
                if len(p.shape) > 1:
                    p.data = random_matrix_fixed_spectrum(p.shape, epsilon)

            #dataset = create_dataset('RANDOM', 1000)
            #data_train, data_test = gp.make_dataset(2000, 500, dimension=input_size, scale=2)

            #model = gp.train_model(model, data_train, data_test, n_epochs=1000)
            #plot_model(model, window_size=10, num_points=1000)

            #compute_lipschitz_approximations(model, random_dataset([input_size], 100, scale=2))

            for p in model.parameters():
                p.requires_grad = False

            # Compute input sizes for all modules of the model
            input_size = get_input_size(random_dataset([input_size], 100, scale=2))
            compute_module_input_sizes(model, input_size)
            res[i] = lipschitz_second_order_ub(model, algo='greedy')

        plt.semilogy(vals, res, label='eigenvalue ratio: {}'.format(epsilon))
    plt.legend()
    plt.xlim(vals[0], vals[-1])
    plt.xlabel('Number of layers')
    plt.ylabel('SeqLip upper bound')
    plt.show()
