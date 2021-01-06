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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, initial_weights: np.array = None, init_bias=None):
        self.shape = (in_features, out_features)
        self.bias = bias

        if initial_weights is not None :
            assert initial_weights.shape == self.shape
            self.weights = initial_weights
        else:
            self.weights = xavier_uniform(self.shape)

        if bias:
            if init_bias is None:

                self.bias = xavier_uniform(out_features)
            else:
                assert init_bias.shape[-1] == out_features
                self.bias = init_bias

    def forward(self, input):
        assert input.shape[-1] == self.shape[0]
        if self.bias:
            return np.dot(input, self.weights) + self.bias
        else:
            return np.dot(input, self.weights)

    def backward(self):
        return


if __name__ == "__main__":
    data = np.array([[1, 2], [2, 3], [3, 4]])
    torch_data = torch.Tensor(data)
    initial_weights = np.ones((2, 4))
    initial_bias = np.ones(4)
    print(initial_weights.shape)
    linear = LinearLayer(2, 4, initial_weights=initial_weights, init_bias=initial_bias, bias=False)
    print(linear.weights)
    output = linear.forward(data)
    print(output)

    with torch.no_grad():
        torch_linear = Linear(2, 4, False)
        torch_linear.weight.fill_(1.)

        output = torch_linear(torch_data)
        print(output)



