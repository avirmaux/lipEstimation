""" Residual neural network toy

MLP with a residual downsampled connexion

EXAMPLE:

    >>> ResidualToy(784, 10, [512, 128, 128])
    >>>
"""

import torch
import torch.nn as nn

class ResidualToy(nn.Module):

    def __init__(self, dim_input, dim_output, layers):
        super(ResidualToy, self).__init__()

        self.downsample = nn.Linear(dim_input, dim_output)

        model = nn.Sequential()
        model.add_module('initial-layer.lin',
                nn.Linear(dim_input, layers[0]))
        model.add_module('initial-layer.relu',
                nn.ReLU())

        for i in range(len(layers)-1):
            model.add_module('layer-{}.lin'.format(i+1),
                    nn.Linear(layers[i], layers[i+1]))
            model.add_module('layer-{}.relu'.format(i+1),
                    nn.ReLU())

        model.add_module('last-layer.lin',
                nn.Linear(layers[-1], dim_output))

        self.model = model


    def forward(self, x):
        return self.downsample(x) + self.model(x)
