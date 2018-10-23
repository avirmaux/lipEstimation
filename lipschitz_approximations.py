import math
import numpy as np
import numpy.random as rd

import scipy.optimize as optimize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from max_eigenvalue import max_eigenvalue, generic_power_method, lipschitz_bn

from lipschitz_utils import *
import experiments.bruteforce_optim as bo


def lipschitz_opt_lb(model, initial_max=None, num_iter=100):
    """ Compute lower bound of the Lipschitz constant with optimization on gradient norm

    INPUTS:
        * `initial_max`: initial seed for the SGD
        * `num_iter`: number of SGD iterations

    If initial_max is not provided, the model must have an input_size attribute
    which consists in a list of torch.Size.
    """
    use_cuda = next(model.parameters()).is_cuda
    if initial_max is None:
        input_size = model.input_sizes[0]
        initial_max = torch.randn(input_size)
    mone = torch.Tensor([-1])
    if use_cuda:
        initial_max = initial_max.cuda()
        mone = mone.cuda()
    v = nn.Parameter(initial_max, requires_grad=True)

    optimizer = optim.Adam([v], lr=1e-3)
    # optimizer = optim.SGD([v], lr=1e-3, momentum=0.9,
    #         nesterov=False)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
            factor=0.5, patience=50, cooldown=10, threshold=1e-6, eps=1e-6,
            verbose=10)

    it = 0

    loss_mean = []
    max_loss = 0

    while it < num_iter:
        # v = initial_max
        optimizer.zero_grad()
        loss = gradient_norm(model, v)**2
        loss.backward(mone)
        optimizer.step()
        loss_mean.append(np.sqrt(loss.data[0]))
        if loss.data[0] > max_loss:
            max_loss = loss.data[0]
        if it%10 == 0:
            print('[{}] {:.4f} (max: {:.4f})'.format(it,
                np.mean(loss_mean), math.sqrt(max_loss)))
            schedule.step(np.mean(loss_mean))
            loss_mean = []

        del loss  # Free the graph
        # v.data.clamp_(-2, 2)

        it += 1

    return gradient_norm(model, v).data[0]


def lipschitz_opt_nm_lb(model, method='Nelder-Mead', maxiter=1e5, verbose=True):
    """ Return Lipschitz lower bound using Nelder-Mead algorithm

    INPUT:
        * `model`
        * `method`: any method accepted by scipy.optimize.minimize
        * `maxiter`: maximum number of iterations
        * `verbose` (default True)

    methods: 'Nelder-Mead' (default), 'Powell'
    """
    from functools import reduce
    from operator import mul
    dim_input = reduce(mul, model.input_sizes[0], 1)
    use_cuda = next(model.parameters()).is_cuda

    # Make model SciPY compliant
    def model_numpy(input):
        x = torch.Tensor(input)
        x = x.view(model.input_sizes[0])
        x = Variable(x, requires_grad=True)
        if use_cuda:
            x = x.cuda()
        grad_norm = - gradient_norm(model, x, requires_grad=False)
        return grad_norm.data[0]

    options = {'disp': verbose,
                'maxiter': maxiter}
    x0 = rd.randn(dim_input)
    if method == 'Nelder-Mead':
        initial_simplex = np.zeros((dim_input+1, dim_input))
        for i in range(dim_input):
            initial_simplex[i, i] = 2
        initial_simplex[-1, :] = rd.rand(dim_input)
        options['initial_simplex'] = initial_simplex

    return -optimize.minimize(model_numpy, x0=x0, method=method, options=options).fun


def lipschitz_opt_annealing(model, *args, **kwds):
    """ Simulated annealing for lower bound of Lipschitz constant

    Basin-hopping algorithm as implemented in SciPY

    WARNING: using CUDA provoks memory errors
    """
    from functools import reduce
    from operator import mul
    dim_input = reduce(mul, model.input_sizes[0], 1)
    use_cuda = next(model.parameters()).is_cuda

    # Make model SciPY compliant
    def model_numpy(input):
        x = torch.Tensor(input)
        x = x.view(model.input_sizes[0])
        x = Variable(x, requires_grad=True)
        if use_cuda:
            x = x.cuda()
        grad_norm = - gradient_norm(model, x, requires_grad=False)
        return grad_norm.data[0]

    x0 = rd.randn(dim_input)
    minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'jac': False}
    return -optimize.basinhopping(func=model_numpy, x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            disp=True, **kwds).fun


def lipschitz_annealing(model, temp=1, step_size=1, batch_size=128, n_iter=64):
    """ Performs a simulated annealing maximisation directly on the model

    Algorithm: use Boltzman-Gibbs energy function, can we do better?

    INPUT:
        * `model`
        * `temp`: initial temperature
        * `step_size`: std of the random walk
        * `batch_size`: number of parallel annealings
        * `n_iter`: number of iterations

    Each time a new maximum is found, slightly reduce the stepsize

    TODO:   * local optimization
    """
    maximum = 0
    n_improve = 0
    t0 = temp
    step0 = step_size
    m = 0
    dim_input = model.input_sizes[0]
    use_cuda = next(model.parameters()).is_cuda
    dim_input = list(dim_input)
    dim_input[0] = batch_size
    dim_input = torch.Size(dim_input)

    batch = Variable(torch.rand(dim_input), requires_grad=True)
    noise = Variable(torch.randn(dim_input), requires_grad=True)
    uniform = torch.Tensor(batch_size)
    model.eval()
    if use_cuda:
        batch = batch.cuda()
        noise = noise.cuda()
        uniform = uniform.cuda()

    for it in range(n_iter):
        temp = t0 / math.log(2+it/2)
        # Random noise
        noise.data.normal_(0, step_size)

        batch = Variable(batch.data, requires_grad=True)
        move = Variable(batch.data + noise.data, requires_grad=True)
        grad_batch = gradient_norm(model, batch, requires_grad=False).data
        grad_move = gradient_norm(model, move, requires_grad=False).data

        tmp_max = grad_batch.max()
        if maximum < tmp_max:
            maximum = tmp_max
            n_improve += 1
            # Slightly reduce stepsize
            # TODO: what's the best way to do it?
            step_size = step0 / math.sqrt(1+n_improve)
            print(' => New maximum: {:.4f} step size: {:.4f}'.format(maximum,
                step_size))
        print('[{}] Best: {:.4f} (moves: {}/{}, T: {:.2f}) (max: {:.4f})'.format(it,
            tmp_max, m, batch_size, temp, maximum))

        # Boltzman-Gibbs transform
        energy = torch.exp((grad_move - grad_batch) / temp)
        # Uniform sample
        uniform.uniform_(0, 1)

        updates = uniform < energy
        m = updates.sum()  # count number of effective moves
        for i in range(batch_size):
            if updates[i]:
                batch.data[i, :] = move.data[i, :]

        # TODO: local optimization

    return maximum


def lipschitz_gsearch_lb(model, constraints, grid_size):
    """ Perform a gridsearch to find Lipschitz lower bound

    INTPUT:
        * `model`: model
        * `constraints`: array of shape [dimensions, 2] given bound of the grid
        for every dimenion
        * `grid_size`: number of points for every dimension

    OUTPUT: maximal value of the gradient at the intersection points

    TODO: batches by the first component

    EXAMPLES:
        If `model` is a function R² -> R then
        >>> grid_search(model, [[-3, 3], [-3, 3]], [100, 100])
    """
    from tqdm import tqdm
    from operator import mul
    from functools import reduce
    from itertools import product
    prod = lambda l: reduce(mul, l, 1)
    maximum = 0
    for inter in tqdm(product(*[np.linspace(cons[0], cons[1], grid_size[i])
        for (i, cons) in enumerate(constraints)]), total=prod(grid_size)):
        x = Variable(torch.Tensor(np.array(inter)).view(1, -1), requires_grad=True)
        grad = gradient_norm(model, x, requires_grad=False).data[0]
        if grad > maximum:
            maximum = grad
    return maximum


def lipschitz_data_lb(model, data, batch_size=128, max_iter=1000):
    """ Compute lower bound of Lipschitz constant on specified data
    """
    if str(data.__class__).find('DataLoader') != -1:
        data_load = data
    else:
        data_load = torch.utils.data.DataLoader(data,
                batch_size=batch_size,
                shuffle=True)  #, num_workers=int(opt.workers))
    use_cuda = next(model.parameters()).is_cuda
    lip_data = 0
    n_iter = 0
    for batch_idx, (real, _) in enumerate(data_load):
        print('batch idx: {}/{} (max: {:.4f})'.format(batch_idx,
            len(data_load), lip_data))
        if use_cuda:
            real = real.cuda()
        # Real data
        real = Variable(real, requires_grad=True)

        lip_data = max(lip_data, gradient_norm(model, real).max())
        n_iter += 1
        if n_iter > max_iter:
            break
    return lip_data


def lipschitz_spectral_ub(model, requires_grad=False):
    """
    Returns the product of spectral radiuses of linear or convolution layer
    of `model`.

    INPUT:
        * `model`: model, must be simple Sequential
    """
    #global lip
    #lip = 1
    use_cuda = next(model.parameters()).is_cuda
    def prod_spec(self, input, output):
        if is_convolution_or_linear(self):
            s, _, _ = generic_power_method(self.forward, self.input_sizes[0],
                    use_cuda=use_cuda)
            # print('{} :: {}'.format(self.__class__.__name__, s.data[0]))
            #global lip
            #lip *= s
            if use_cuda:
                s = s.cpu()
            self.spectral_norm = s
            # print(self.__class__.__name__, s)
        if is_batch_norm(self):
            # One could have also used generic_power_method
            s = lipschitz_bn(self)
            self.spectral_norm = s
            # print(self.__class__.__name__, s)
    execute_through_model(prod_spec, model)
    # Return product of Lipschitz constants
    # WARNING: does not work with multiple inputs per layer
    lipschitz = Variable(torch.Tensor([1]), requires_grad=requires_grad)
    for p in model.modules():
        if hasattr(p, 'spectral_norm'):
            lipschitz = lipschitz * p.spectral_norm  # not in-place for grad
    return lipschitz


def lipschitz_frobenius_ub(model, requires_grad=False):
    """
    Returns the product of Frobenius norms of each matrices in
    each layer.  It is an upper-bound of the Lipschitz constant of the net.

    TODO: * incorporate the bias
          * get ride of global variable

    WARNING: for convolutions, we now take the frobenius norm
             of the parameter matrix, which is not correct...
    """
    #global lip
    #lip = 1
    def prod_frob(self, input, output):
        print(self.__class__.__name__)
        if is_convolution_or_linear(self):
            p = list(self.parameters())[0]
            #global lip
            #lip *= torch.norm(p, p=2).cpu()
            self.frob_norm = torch.norm(p, p=2).cpu()
    execute_through_model(prod_frob, model, backwards=True)
    #res = lip
    #del lip
    #return res
    frob = Variable(torch.Tensor([1]), requires_grad=requires_grad)
    for p in model.modules():
        if hasattr(p, 'frob_norm'):
            frob = frob * p.frob_norm  # not in-place for grad
    return frob


def lipschitz_second_order_ub(model, algo='greedy'):
    '''
    Computes an upper bound by maximizing over \sigma inside the sum...

    INPUT:
        * `model`
        * `algo` may be {'greedy', 'bfgs'}

    TODO: * check if correct!
          * extend to other than input_size=2 and ouput_size=1...

    WARNING: for convolutions, we now take the frobenius norm
             of the parameter matrix, which is not correct...
    '''

    def affine_layers_aux(self, input, output):
        # print(self.__class__.__name__)
        if is_convolution_or_linear(self):
            # calling self.forward(v) instead of self(v) prevent hooks from
            # being called and ending in an infinite loop.
            global var_through_lin, u_prev, s_prev
            #print('var_', var_through_lin)
            p = list(self.parameters())[0]
            u, s, v = np.linalg.svd(p.data.cpu().numpy())
            v = v.transpose()
            #print('s', s)
            #print('u', u.shape)
            #print('v', v.shape)
            if s.shape[0] > 1 and u_prev is not None:
                print('ratio s', s[1] / s[0])
                #M = compute_abs_prod(v, u_prev)[0, 0]**2
                #print('M', M)
                #r = s[1] / s[0]
                #r_prev = s_prev[1] / s_prev[0]
                #r_max = max(r, r_prev)
                #factor = ((1 - r_max) * M + r + r_prev)**0.5
                #print('factor', factor)
                #var_through_lin *= (s[1] * s_prev[1])**0.5 * min(1, factor)

                print('factor abs prod:', compute_abs_prod(u_prev, v)[0,0])
                s2 = resize_with_zeros(s, v.shape[0])
                s_prev2 = resize_with_zeros(s_prev, u_prev.shape[0])
                # factor, a = bo.optim(np.diag(s2**0.5) @ v, u_prev @ np.diag(s_prev2**0.5))
                if algo == 'bfgs':
                    factor, a = bo.optim_approx(np.diag(s2**0.5) @ v,
                            u_prev @ np.diag(s_prev2**0.5), verbose=False)
                elif algo == 'exact':
                    factor, a = bo.optim_bf(np.diag(s2**0.5) @ v,
                            u_prev @ np.diag(s_prev2**0.5), verbose=True)
                else:
                    factor, a = bo.optim_greedy(np.diag(s2**0.5) @ v,
                            u_prev @ np.diag(s_prev2**0.5), verbose=False)
                factor /=  (s[0] * s_prev[0])**0.5
                # spectral *= tmp_spec
                print('factor', factor)
                var_through_lin *= factor
            #print(s[0]**0.5)
            #print(var_through_lin)
            var_through_lin *= (s[0] * s_prev[0])**0.5
            #print(var_through_lin)
            u_prev = u
            s_prev = s
            #print('var_end', var_through_lin)

    global var_through_lin, u_prev, s_prev
    var_through_lin = 1
    u_prev = None
    s_prev = [1]
    execute_through_model(affine_layers_aux, model)
    #print('var_final', var_through_lin)
    #print('s_prev', s_prev)
    res = s_prev[0]**0.5 * var_through_lin
    #print('res', res)
    del var_through_lin, u_prev, s_prev
    #print('res', res)
    return res

def resize_with_zeros(x, size):
    ''' Resizes the vector with trailing zeros if necessary.'''
    y = np.zeros(size)
    n = min(x.shape[0], size)
    y[:n] = x[:n]
    return y


def compute_pos_prod(u, v):
    '''Scalar product of two vectors keeping only positive terms'''
    n = u.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.sum(np.maximum(u[:, i] * v[:, j], np.zeros(n)))
    return M


def compute_abs_prod(u, v):
    '''Scalar product of two vectors keeping only positive or negative terms'''
    return np.maximum(compute_pos_prod(u, v), compute_pos_prod(-u, v))
