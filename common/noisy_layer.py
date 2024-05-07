import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):
    """
    Noisy weights in a linear layer from the NoisyNets paper
    """
    def __init__(self, input_size, output_size, sigma_init=0.5):
        """
        y = Wx+b, except each element of W and b is noisy
        with mean (mu) and variance (sigma^2) that are learned.

        Zero-mean noise is injected through the epsilon parameter.
        The current noise used in the model is saved to a buffer,
        so that when the model is saved, the noise is also saved
        """
        super(NoisyLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sigma_init = sigma_init
        self.training_noise = True

        ## Noisy Weights
        self.weights_mu = nn.Parameter(torch.Tensor(output_size, input_size))
        self.weights_sigma = nn.Parameter(torch.Tensor(output_size, input_size))
        self.register_buffer("weight_epsilon", torch.Tensor(output_size, input_size))
        
        ## Noisy Bias
        self.bias_mu = nn.Parameter(torch.Tensor(output_size))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_size))
        self.register_buffer("bias_epsilon", torch.Tensor(output_size))

        ## Reset the weights, bias, and noise
        self.reset_parameters()
        self.reset_noise()

    def train_noise(self):
        self.training_noise = True

    def eval_noise(self):
        self.training_noise = False

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.input_size)
        self.weights_mu.data.uniform_(-mu_range, mu_range)
        self.weights_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in  = self.scale_noise(self.input_size)
        epsilon_out = self.scale_noise(self.output_size)

        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(self.scale_noise(self.output_size))

    def forward(self, inputs):
        ## Note: this depends on whether train() or eval() is called
        if self.training or self.training_noise:
            return F.linear(inputs, self.weights_mu + self.weights_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inputs, self.weights_mu, self.bias_mu)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"NoisyLinear({self.input_size}, {self.output_size}, {self.sigma_init})"

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
