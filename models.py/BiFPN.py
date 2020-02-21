import torch
from torch import nn
import torch.functional as F
import numpy as np
import math


class BiFPNModule(nn.Module):
    def __init__(self, max_input_size, num_layer=5):
        if max_input_size <= 2 ** (num_layer-1) or not math.log2(max_input_size).is_integer():
            raise ValueError("invalid input size to BiFPNModule!")
        self.input_sizes = [max_input_size / (2 ** x) for x in range(num_layer)]
        self.weights = nn.Parameter(torch.rand((num_layer, 5)), requires_grad = TRUE)
        self.output_convs = nn.ModuleList()
        self.intermediate_convs = nn.ModuleList()
        for i in range(num_layer):
            intermediate = 