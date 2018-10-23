""" Generic ReLU MLP class

EXAMPLE:

    >>> MLP(784, 10, [512, 128, 128])
    >>>
"""

import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, dim_input, dim_output, layers):
        super(MLP, self).__init__()

        model = nn.Sequential()
        model.add_module('initiallin',
                nn.Linear(dim_input, layers[0]))
        model.add_module('initialrelu',
                nn.ReLU())

        for i in range(len(layers)-1):
            model.add_module('layer-{}lin'.format(i+1),
                    nn.Linear(layers[i], layers[i+1]))
            model.add_module('layer-{}relu'.format(i+1),
                    nn.ReLU())

        model.add_module('finallin',
                nn.Linear(layers[-1], dim_output))

        self.model = model

    def forward(self, x):
        return self.model(x)
