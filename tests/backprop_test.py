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
        num_data = 30
        num_input_feature = 4
        num_hidden_feature = 12
        num_out_dimension = 7
        learning_rate = 0.3

        x = torch.randn(num_data, num_input_feature)
        y = torch.randn(num_data, num_out_dimension)
        print(x)
        print(y)

        # model
        model = torch.nn.Sequential(
            Linear(num_input_feature, num_hidden_feature),
            Linear(num_hidden_feature, num_out_dimension),
            torch.nn.Sigmoid()
        )

        # loss and optimizer
        print('\n\n===================================================')
        print('====================[TORCH]========================')
        print('===================================================\n\n')

        print('[linear weight]\n', model[0].weight)
        print('[linear weight]\n', model[0].bias)

        loss_fn = torch.nn.MSELoss(reduction='mean')
        y_pred = model(x)
        y_pred.retain_grad()

        loss = loss_fn(y_pred, y)
        loss.backward()

        print('[torch loss]', loss)
        print('[linear grad]\n', model[0].weight.grad.data)
        y_pred_grad_torch = y_pred.grad.data.numpy()
        print('[y_pred grad]\n', y_pred_grad_torch, y_pred.shape)
        optimizer = optim.SGD(lr=learning_rate, params=model.parameters())

        print('[linear weight after]\n', model[0].weight)

        print('\n\n===================================================')
        print('====================[CUSTOM]=======================')
        print('===================================================\n\n')

        initial_weights = model[0].weight.detach().numpy().transpose()
        initial_bias = model[0].bias.detach().numpy()

        initial_weights_1 = model[1].weight.detach().numpy().transpose()
        initial_bias_1 = model[1].bias.detach().numpy()

        x = x.numpy()
        y = y.numpy()

        # layer initialization

        linearlayer_custom_0 = LinearLayer(num_input_feature, num_hidden_feature,
                                           initial_weights=initial_weights,
                                           initial_bias=initial_bias)

        linearlayer_custom_1 = LinearLayer(num_hidden_feature, num_out_dimension,
                                           initial_weights=initial_weights_1,
                                           initial_bias=initial_bias_1)
        sigmoid_custom = SigmoidLayer(x.shape)

        # feed forward

        print('[0] initial weight\n', initial_weights)
        hidden = linearlayer_custom_0.forward(x)

        out = linearlayer_custom_1.forward(hidden)

        y_pred_custom = sigmoid_custom.forward(out)
        loss_custom = np.mean(np.power(y - y_pred_custom, 2))

        print('custom loss', loss_custom)

        self.assertTrue(loss_custom - loss < 1e-4)

        gradient_from_loss = 2 * (y_pred_custom - y) / (y.shape[1] * y.shape[0])

        print('[1] gradient_from_loss\n', gradient_from_loss.shape, gradient_from_loss)

        dout = sigmoid_custom.backward(gradient_from_loss)

        print(dout.shape)
        print('[2] sigmoid grad1 mean', dout)

        dweight1 = linearlayer_custom_1.backward(dout)

        torch_grad_1 = model[1].weight.grad.data.numpy()
        custom_grad_1 = linearlayer_custom_1.grad.transpose()

        print('shape', torch_grad_1.shape, custom_grad_1.shape)

        dhidden = np.dot(dout, linearlayer_custom_1.weights.transpose())
        assert (hidden.shape == dhidden.shape)

        dweight0 = linearlayer_custom_0.backward(dhidden)

        torch_grad = model[0].weight.grad.data.numpy()
        custom_grad = linearlayer_custom_0.grad.transpose()

        print('\n\n[3 linear layer gradient]\n')
        print(model[0].weight.grad.data)
        print(linearlayer_custom_0.grad.transpose())
        print('DEVIDE', torch_grad / custom_grad)

        epsilon = 1e-4
        self.assertTrue(np.alltrue(torch_grad - custom_grad < epsilon))

        new_weights_custom = linearlayer_custom_0.weights - (learning_rate * dweight0)
        new_weights_torch = initial_weights.transpose() - (learning_rate * model[0].weight.grad.data.detach().numpy())

        optimizer.step()
        self.assertTrue(np.all(new_weights_torch - model[0].weight.detach().numpy()) < 1e-4)
        self.assertTrue(np.all(new_weights_torch - new_weights_custom.transpose()) < 1e-4)

    def test_linear_batch_step(self):
        pass


if __name__ == '__main__':
    unittest.main()
