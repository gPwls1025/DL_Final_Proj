from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

IMAGE_SIZE = 64
TOL = 1e-6

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class CNN(nn.Module):
    def __init__(self, channel_sizes, out_dim, pool_size, kernel_size, final_activation):
        super(CNN, self).__init__()
        assert abs(len(channel_sizes) - np.log(IMAGE_SIZE) / (np.log(pool_size))) < TOL, "Incompatible channel sizes and pool sizes!"

        padding = int((kernel_size - 1) / 2)
        self.final_activation = final_activation
        self.conv_modules = nn.ModuleList()

        for idx in range(len(channel_sizes)):
            self.conv_modules.append(nn.Conv2d(
                in_channels=(1 if idx == 0 else channel_sizes[idx-1]),
                out_channels=channel_sizes[idx],
                kernel_size=kernel_size,
                padding=padding
            )
        )
        
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.fc = nn.Linear(channel_sizes[-1], out_dim)
    
    def forward(self, x):
        for conv_layer in self.conv_modules:
            x = torch.relu(conv_layer(x))
            x = self.pool(x)

        x = x[:,:,0,0]
        return self.final_activation(self.fc(x))


class MLP(nn.Module):
    def __init__(self, layer_sizes, final_activation, leaky_relu_mult):
        super(MLP, self).__init__()

        self.final_activation = final_activation
        self.layers = nn.ModuleList()
        self.leaky_relu_mult = leaky_relu_mult

        for idx in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        return self.final_activation(self.layers[-1](x))


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
