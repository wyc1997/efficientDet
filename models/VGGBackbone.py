import torch
from torch import nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, number, in_channel, out_channel, kernel_size, stride, padding, activation, maxpooling=None):
        super(ConvModule, self).__init__()
        self.in_channel = in_channel
        self.layers = nn.Sequential()
        self.layers.add_module(f"conv{kernel_size}_{number}", nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
        self.layers.add_module(f"act_{activation}_{number}", activation)
        if maxpooling != None:
            self.layers.add_module(f"maxpool_{maxpooling}", nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.layers(x)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.convs = nn.ModuleList()
        x = 0
        self.convs.append(ConvModule(x, 3, 64, 3, 1, 1, nn.ReLU()))
        x += 1
        self.convs.append(ConvModule(x, 64, 64, 3, 1, 1, nn.ReLU(), maxpooling=0))
        x += 1
        self.convs.append(ConvModule(x, 64, 128, 3, 1, 1, nn.ReLU()))
        x += 1
        self.convs.append(ConvModule(x, 128, 128, 3, 1, 1, nn.ReLU(), maxpooling=1))
        x += 1
        while x < 8:
            if x == 4:
                self.convs.append(ConvModule(x, 128, 256, 3, 1, 1, nn.ReLU()))
            elif x == 7:
                self.convs.append(ConvModule(x, 256, 256, 3, 1, 1, nn.ReLU(), maxpooling=2))
            else:
                self.convs.append(ConvModule(x, 256, 256, 3, 1, 1, nn.ReLU()))
            x += 1
        while x < 16:
            if x == 8:
                self.convs.append(ConvModule(x, 256, 512, 3, 1, 1, nn.ReLU()))
            elif x == 11 or x == 15:
                self.convs.append(ConvModule(x, 512, 512, 3, 1, 1, nn.ReLU(), maxpooling=3))
            else:
                self.convs.append(ConvModule(x, 512, 512, 3, 1, 1, nn.ReLU()))
            x += 1
        

    def forward(self, x):
        output = []
        for i in range(16):
            x = self.convs[i](x)
            if i == 1 or i == 3 or i == 7 or i == 11 or i == 15:
                output.append(x)
        output.reverse()
        return output