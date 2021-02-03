import numpy as np
import torch
from torch.nn import Linear
import math


def xavier_uniform(size):
    np.random.seed(0)
    scale = 1 / max(1., (2 + 2) / 2.)
    limit = math.sqrt(3.0 * scale)
    weights = np.random.uniform(-limit, limit, size=size)
    return weights


class LinearLayer:

    def __init__(self, in_features: int, out_features: int,
                 bias_exist: bool = True,
                 initial_weights: np.array = None,
                 initial_bias=None):
        self.shape = (in_features, out_features)
        self.bias_exist = bias_exist

        if initial_weights is not None:
            assert initial_weights.shape == self.shape
            self.weights = initial_weights
        else:
            self.weights = xavier_uniform(self.shape)

        if bias_exist:
            if initial_bias is None:
                self.bias = xavier_uniform(out_features)
            else:
                assert initial_bias.shape[-1] == out_features
                self.bias = initial_bias

    def forward(self, input):
        assert input.shape[-1] == self.shape[0]
        self.input = input

        if self.bias_exist:
            return np.dot(input, self.weights) + self.bias
        else:
            return np.dot(input, self.weights)

    def backward(self, gradient_from_forward=None):
        if gradient_from_forward is None:
            gradient_from_forward = np.ones((1, self.shape[-1]))  # (1, out_featues)
        else:
            print('gradient_from_forward shape:', gradient_from_forward.shape, self.input.shape)
            assert gradient_from_forward.shape[1] == self.shape[1], \
                f'{gradient_from_forward.shape[1]}!={self.shape[1]}'

        self.grad = np.zeros(self.shape)
        if self.bias_exist:
            self.grad_bias = np.zeros(self.bias.shape)

        # shape: (in_features, out_features)
        self.grad = np.dot(self.input.transpose(), gradient_from_forward)
        return self.grad


class SigmoidLayer():
    def __init__(self, input_shape):
        self.shape = input_shape[:1]

    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, gradient_from_forward=None):
        if gradient_from_forward is not None:
            print('backward sigmoid..', gradient_from_forward.shape, self.shape)

        if gradient_from_forward is None:
            gradient_from_forward = 1
        else:
            pass

        current_gradient = self.forward(self.input) * (1 - self.forward(self.input))
        print('sigmoid backward.. current gradient shape', current_gradient.shape)
        self.grad = gradient_from_forward * current_gradient
        return self.grad


def MSELoss(y, y_pred):
    N, _, _ = y.shape
    N_pred, _, _, = y_pred.shape

    assert N == N_pred

    return np.mean(y - y_pred, axis=0)


if __name__ == "__main__":
    data = np.array([[1, 2], [2, 3], [3, 4]])
    torch_data = torch.Tensor(data)
    initial_weights = np.ones((2, 4))
    initial_bias = np.ones(4)

    print(initial_weights.shape)
    linear = LinearLayer(2, 4,
                         initial_weights=initial_weights,
                         initial_bias=initial_bias,
                         bias_exist=False)

    print(linear.weights)
    output = linear.forward(data)
    print(output)

    with torch.no_grad():
        torch_linear = Linear(2, 4, False)
        torch_linear.weight.fill_(1.)
        output = torch_linear(torch_data)
        print(output)
