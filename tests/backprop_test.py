import unittest
import torch
import numpy as np
from torch.nn import Linear
from backprop import LinearLayer, xavier_uniform


class TestBackProp(unittest.TestCase):

    def test_linear_layer_feed_forward(self):
        num_hidden_layer = 4
        num_input_data_feature = 2
        data = np.array([[1, 2], [2, 3], [3, 4]])

        self.assertEqual(data.shape[-1], num_input_data_feature)
        torch_data = torch.Tensor(data)

        initial_weights = xavier_uniform((2, num_hidden_layer))
        initial_bias = np.ones(num_hidden_layer)

        linear = LinearLayer(num_input_data_feature, num_hidden_layer, initial_weights=initial_weights,
                             init_bias=initial_bias, bias=False)
        output = linear.forward(data)

        with torch.no_grad():
            torch_linear = Linear(num_input_data_feature, num_hidden_layer, False)
            torch_linear.weight = torch.nn.Parameter(
                torch.Tensor(initial_weights.transpose()))

            output_torch = torch_linear(torch_data)
            epsilon = 1e-5

            self.assertTrue(np.alltrue(output - output_torch.numpy() < epsilon))

    def test_linear_layer_back_prop(self):
        num_input_feature = 3
        num_hidden_feature = 4
        data = np.random.random((1, 3))
        data = np.array(data).astype(np.float32)
        data_torch = torch.from_numpy(data)

        data_torch.requires_grad = True

        torch_linear = Linear(num_input_feature, num_hidden_feature)

        hidden = torch_linear(data_torch)
        output = torch.sum(hidden)

        output.backward()
        print('data', data)
        print('weight', torch_linear.weight)
        print('output', output)
        print('grad', data_torch.grad)
        print(hidden.grad)
        print(torch_linear.weight.grad)


if __name__ == '__main__':
    unittest.main()
