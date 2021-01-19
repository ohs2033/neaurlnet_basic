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
        # parameters
        num_data = 10000
        num_input_feature = 3
        num_out_dimension = 4
        lr = 1.0

        x = torch.randn(num_data, num_input_feature)
        y = torch.randn(num_data, num_out_dimension)

        # model
        model = torch.nn.Sequential(
            Linear(num_input_feature, num_out_dimension),
            torch.nn.Sigmoid()
        )

        # loss and optimizer
        print('[linear weight]\n', model[0].weight)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        print('[torch loss]', loss)
        loss.backward()
        print('[linear grad]\n', model[0].weight.grad.data)
        print('[y_pred grad]', y_pred.grad)

        # optimizer.step()
        print('[linear weight after]\n', model[0].weight)

        initial_weights = model[0].weight.detach().numpy().transpose()
        initial_bias = model[0].bias.detach().numpy()


        x = x.numpy()

        # layer initialization
        linearlayer_custom = LinearLayer(num_input_feature, num_out_dimension,
                                  initial_weights=initial_weights,
                                  initial_bias=initial_bias)

        sigmoid_custom = SigmoidLayer(x.shape)

        # feed forward
        hidden = linearlayer_custom.forward(x)
        y_pred_custom = sigmoid_custom.forward(hidden)
        y_numpy = y.numpy()
        print(y_pred_custom.shape)
        print(y_numpy.shape)
        loss = np.mean(np.power(y.numpy() - y_pred_custom, 2))
        print('custom loss', loss)
        gradient_from_loss = np.mean(2 * (y_pred_custom - y_numpy),axis=0)

        print('gradient_from_loss', gradient_from_loss.shape, gradient_from_loss)

        grad1 = sigmoid_custom.backward(gradient_from_loss)
        grad1_mean = np.sum(grad1, axis=0)
        grad1_mean = np.reshape(grad1_mean, (1, -1))

        print(grad1_mean.shape)

        print('sigmoid grad1 mean', grad1_mean)

        linearlayer_custom.backward(grad1_mean)
        print('custom gradient')
        print(linearlayer_custom.grad.transpose())

        # custom_model =


if __name__ == '__main__':
    unittest.main()
