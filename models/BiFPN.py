import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class BiFPNModule(nn.Module):
    def __init__(self, max_input_size, output_channel, num_layer=5, eps=0.0001):
        super().__init__()
        if max_input_size <= 2 ** (num_layer-1) or not math.log2(max_input_size).is_integer():
            raise ValueError("invalid input size to BiFPNModule!")
        self.weights = nn.Parameter(torch.rand((num_layer, 5)), requires_grad = True)
        self.num_layer = num_layer
        self.eps = eps
        self.output_channel = output_channel
        self.output_convs = nn.ModuleList()
        self.intermediate_convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(output_channel)
        for i in range(num_layer):
            if i == 0 or i == num_layer - 1:
                self.intermediate_convs.append(None)
            else:
                intermediate = nn.Conv2d(output_channel, output_channel, 3, 1, padding=1)
                self.intermediate_convs.append(intermediate)
        for i in range(num_layer):
            out = nn.Conv2d(output_channel, output_channel, 3, 1, padding=1)
            self.output_convs.append(out)

    
    def forward(self, inputs):
        # print(inputs)
        # print(len(inputs))
        intermediate_out = []
        intermediate_out.append(inputs[0])
        weights = self.relu(self.weights)
        weights /= torch.sum(weights, dim=1) + self.eps #normalizing

#       top down path
        for i in range(self.num_layer):
            if self.intermediate_convs[i] == None:
                continue
            conv_input = (weights[i, 0] * inputs[i] + weights[i, 1] * 
                          F.interpolate(inputs[i-1], scale_factor=2)) /(weights[i, 0] + weights[i, 1] + self.eps)
            conv_out = self.bn(self.intermediate_convs[i](conv_input))
            intermediate_out.append(self.relu(conv_out))

        intermediate_out.append(inputs[self.num_layer - 1])
#       bottom up path
        outputs = [None] * self.num_layer
        for i in range(self.num_layer - 1, -1, -1):
            if i == (self.num_layer - 1):
                conv_input = (weights[i, 3] * intermediate_out[i] + 
                            weights[i, 4] * F.interpolate(intermediate_out[i - 1], scale_factor=2))/(weights[i, 3] + weights[i, 4] + self.eps)
                conv_out = self.bn(self.output_convs[i](conv_input))
                outputs[i] = self.relu(conv_out)
            elif i == 0:
                conv_input = (weights[i, 3] * intermediate_out[i] + weights[i, 4] * 
                                F.max_pool2d(intermediate_out[i+1], kernel_size=2, stride=2))/(weights[i, 3] + weights[i, 4] + self.eps)
                conv_out = self.bn(self.output_convs[i](conv_input))
                outputs[i] = self.relu(conv_out)
            else:
                conv_input = (weights[i, 2] * inputs[i] + weights[i, 3] * intermediate_out[i] + 
                            weights[i, 4] * F.max_pool2d(intermediate_out[i + 1], kernel_size=2, stride=2))/(weights[i, 2] + weights[i, 3] + weights[i, 4] + self.eps)
                conv_out = self.bn(self.output_convs[i](conv_input))
                outputs[i] = self.relu(conv_out)
        return outputs

class BiFPN(nn.Module):
    def __init__(self, num_modules, input_sizes, in_channels, out_channel, num_layer=5):
        super().__init__()
        self.num_modules = num_modules
        self.num_layer = num_layer
        self.out_channel = out_channel
        self.in_channels = in_channels
        self.bifpn_modules = nn.Sequential()
        for i in range(num_modules):
            self.bifpn_modules.add_module(f"bifpn{i}", BiFPNModule(
                max_input_size = max(input_sizes),
                output_channel = self.out_channel,
                num_layer=self.num_layer
            ))
        self.convs = nn.ModuleList()
        if len(self.in_channels) != self.num_layer:
            raise(ValueError("number of input channels not equal to num_layer"))
        for i in range(num_layer):
            conv = nn.Conv2d(in_channels[i], self.out_channel, kernel_size=3, stride=1, padding=1)
            self.convs.append(conv)

    def forward(self, inputs):
        processed = []
        for i, input_feature in enumerate(inputs):
            x = self.convs[i](input_feature)
            processed.append(x)
        outputs = self.bifpn_modules(processed)
        return outputs




            