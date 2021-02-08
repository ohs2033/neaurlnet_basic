import torch
import numpy as np
import unittest


class TestConvolutionNeuralNetwork(unittest.TestCase):

    def test_cnn_forward(self):
        epsilon = 1e-6

        batch_size = 10
        input_image_width = 32
        input_iamge_height = input_image_width
        num_input_layer = 3

        num_hidden_layer = 5

        padding = 1
        stride = 1
        kernel_size = 3

        padding_mode = 'zeros'
        padding_value = 0 if padding_mode == 'zeros' else None

        random_value = np.random.random((batch_size, num_input_layer, input_image_width, input_iamge_height))
        input_tensor = torch.Tensor(random_value)

        conv2d = torch.nn.Conv2d(num_input_layer, num_hidden_layer, kernel_size=(kernel_size, kernel_size),
                                 stride=stride, padding=padding, padding_mode=padding_mode)

        output = conv2d.forward(input_tensor)

        torch_weight = conv2d.weight.data.detach().numpy()
        torch_bias = conv2d.bias.data.detach().numpy()

        basic_step_size = input_image_width - kernel_size + 1  # 30?
        padded_step_size = basic_step_size + 2 * padding  # 32

        output_width = padded_step_size

        output_featuremap = np.zeros((batch_size, num_hidden_layer, output_width, output_width))

        for batch in range(0, batch_size):
            data = random_value[batch]
            padded_data = np.pad(data, ((0, 0), (padding, padding), (padding, padding)), 'constant',
                                 constant_values=padding_value)
            for row in range(0, padded_step_size):
                for col in range(0, padded_step_size):
                    for output_layer in range(num_hidden_layer):
                        weight = torch_weight[output_layer]
                        bias = torch_bias[output_layer]
                        target = padded_data[:, row:row + kernel_size, col:col + kernel_size]
                        outs = np.sum(np.multiply(target, weight)) + bias
                        output_featuremap[batch, output_layer, row, col] = outs

        torch_output = output.detach().numpy()

        self.assertEqual(output_featuremap.shape, torch_output.shape)
        self.assertTrue(np.alltrue(output_featuremap - torch_output < epsilon))

        pass

    def test_cnn_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
