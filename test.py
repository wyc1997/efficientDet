import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from models.VGGBackbone import *

import numpy as np

def main():
    model = VGG16()
    summary(model, (3, 512, 512))
    

if __name__ == '__main__':
    main()
        