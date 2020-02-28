import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm

from models.VGGBackbone import *
from models.BiFPN import *

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = VGG16()
    model.to(device)
    summary(model, (3, 512, 512))
    output = model(torch.rand((1, 3, 512, 512)).to(device))
    output_sizes = [out.shape[2] for out in output]
    output_channels = [out.shape[1] for out in output]

    BiFPN_layer = BiFPN(3, output_sizes, output_channels, 128)
    BiFPN_layer.to(device)
    BiFPN_output = BiFPN_layer(output)
    for i,x in enumerate(BiFPN_output):
        print(f"layer{i}", x.shape)
    

if __name__ == '__main__':
    main()
        