import torch
from torch import nn
import torch.functional as F
import numpy as np
import math


class BiFPNModule(nn.Module):
    def __init__(self, max_input_size, input_channels, output_channels, num_layer=5):
        if max_input_size <= 2 ** (num_layer-1) or not math.log2(max_input_size).is_integer():
            raise ValueError("invalid input size to BiFPNModule!")
        self.input_sizes = [max_input_size / (2 ** x) for x in range(num_layer)]
        self.weights = nn.Parameter(torch.rand((num_layer, 5)), requires_grad = TRUE)
        self.input_channels = input_channels
        self.output_convs = nn.ModuleList()
        self.intermediate_convs = nn.ModuleList()
        for i in range(1, num_layer - 1):
            intermediate = nn.Conv2d(input_channels[i], output_channels, 3, 1, padding=1)
            intermediate_act = nn.LeakyReLU(-0.01)
            self.intermediate_convs.append(intermediate)
            self.intermediate_convs.append(intermediate_act)
        for i in range(num_layer - 1, -1, -1):
            out = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
            out_act = nn.LeakyReLU(-0.01)
