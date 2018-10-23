""" Gaussian processes experimentations
"""
import numpy as np
import numpy.random as rd
from sklearn.gaussian_process import GaussianProcessRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

def make_dataset(n_train, n_test, dimension=2, scale=4, x=[], y=[]):
    """ Return a dataset obtained from a Gaussian process in dimension
    `dimension`

    INPUT:
        * `dimension`: dimension
        * `prior` must be a list of points of the form of a NumPY array
        * `n_train`: number of points in the train set
        * `n_test`: number of points in the test set


    EXAMPLE:
        >>> x = np.array_([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> y = np.array([1, -1, -1, 1])
        >>> x_train, y_train, x_test, y_test = make_dataset(10, 5, x, y)
    """
    gp = GaussianProcessRegressor()
    if len(x) > 0:
        gp.fit(x, y)

    # Uniform sample points in the cube [-1, 1]
    x_train = torch.Tensor(scale * rd.randn(n_train, dimension))
    x_test  = torch.Tensor(scale * rd.randn(n_test, dimension))
    y_train = torch.Tensor(gp.sample_y(x_train))
    y_test  = torch.Tensor(gp.sample_y(x_test))
    return x_train, y_train

    data_train = torch.utils.data.TensorDataset(x_train, y_train)
    data_test = torch.utils.data.TensorDataset(x_test, y_test)
    return data_train, data_test


def train_model(model, data_train, data_test, n_epochs=10, batch_size=128):
    """ Experimentations on the GP generated data
    """

    train_data = data.DataLoader(data_train,
            batch_size=batch_size,
            shuffle=True)
    # test_data = data.DataLoader(data_test,
    #         batch_size=batch_size,
    #         shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
            factor=0.3, min_lr=1e-7, patience=3000, verbose=True)
    loss_fn = nn.MSELoss()

    epoch = 0
    while epoch < n_epochs:

        for (idx, (x, y)) in enumerate(train_data):
            x, y = Variable(x), Variable(y)
            model.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if idx % 100 == 0:
                print('[{}: {}/{}] Loss: {:.4f}'.format(epoch, idx,
                    len(train_data), loss.data.item()))

        epoch += 1
    return model

