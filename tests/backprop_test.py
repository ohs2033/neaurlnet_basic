import unittest
import torch
import numpy as np
from torch.nn import Linear
import torch.optim as optim
from backprop import LinearLayer, SigmoidLayer, xavier_uniform


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
                             initial_bias=initial_bias, bias_exist=False)
        output = linear.forward(data)

        with torch.no_grad():
            torch_linear = Linear(num_input_data_feature, num_hidden_layer, False)
            torch_linear.weight = torch.nn.Parameter(
                torch.Tensor(initial_weights.transpose()))

            output_torch = torch_linear(torch_data)
            epsilon = 1e-5

            self.assertTrue(np.alltrue(output - output_torch.numpy() < epsilon))

    def test_linear_layer_back_prop(self):
        epsilon = 1e-4
        num_input_feature = 3
        num_hidden_feature = 4
        num_data = 1
        data = np.random.random((num_data, num_input_feature))
        data = np.array(data).astype(np.float32)

        data_torch = torch.from_numpy(data)
        data_torch.requires_grad = True

        torch_linear = Linear(num_input_feature, num_hidden_feature)

        hidden = torch_linear(data_torch)
        torch_output = torch.sum(hidden)
        torch_output.backward()

        # print('data', data)
        # print('weight', torch_linear.weight)
        # print('output', 1)
        # print('grad', data_torch.grad)
        # print(hidden.grad)
        torch_grad = torch_linear.weight.grad.data

        # print('torch linear layer weight!', torch_linear.weight.grad)
        print('\n<< 1 >> torch weight grad!\n', torch_grad.numpy().transpose())

        initial_weights = torch_linear.weight.detach().numpy().transpose()
        initial_bias = torch_linear.bias.detach().numpy()


        linearlayer = LinearLayer(num_input_feature, num_hidden_feature, initial_weights=initial_weights,
                                  initial_bias=initial_bias)
        output = linearlayer.forward(data)
        linearlayer.backward()
        print('<< 2 >> custom layer weight grad!\n', linearlayer.grad)
        diff = linearlayer.grad - torch_grad.numpy().transpose()
        self.assertTrue(np.alltrue(np.abs(diff) < epsilon))

    def test_multilayer_backprop(self):
        epsilon = 1e-4
        num_input_feature = 5
        num_hidden_feature = 4
        data = np.random.random((10, num_input_feature))

        data = np.array(data).astype(np.float32)

        # torch
        data_torch = torch.from_numpy(data)
        data_torch.requires_grad = True
        torch_linear = Linear(num_input_feature, num_hidden_feature)
        out1_torch = torch_linear(data_torch)
        out2_torch = torch.sigmoid(out1_torch)
        out3_torch = torch.sum(out2_torch)
        out3_torch.backward()

        print('weight gradient from torch:')
        print(torch_linear.weight.grad)
        print('weight gradient for sigmoid', out1_torch.grad)
        # custom gradient
        initial_weights = torch_linear.weight.detach().numpy().transpose()
        initial_bias = torch_linear.bias.detach().numpy()

        # layer initialization
        linearlayer = LinearLayer(num_input_feature, num_hidden_feature, initial_weights=initial_weights,
                                  initial_bias=initial_bias)

        sigmoid = SigmoidLayer(data.shape)

        # feed forward
        hidden = linearlayer.forward(data)
        out = sigmoid.forward(hidden)
        grad1 = sigmoid.backward()

        grad1_mean = np.mean(grad1, axis=0)
        grad1_mean = np.reshape(grad1_mean, (1, -1))

        print(grad1_mean.shape)

        print('sigmoid grad1 mean', grad1_mean)

        linearlayer.backward(grad1_mean)

        print('custom gradient')
        print(linearlayer.grad.transpose())

    def test_linear_batch_gradient(self):
        num_input_feature = 3
        num_hidden_feature = 10
        model = torch.nn.Sequential([])

        optimizer = optim.SGD()

if __name__ == '__main__':
    unittest.main()
