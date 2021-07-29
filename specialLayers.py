import torch
import torch.nn as nn
import numpy as np
import math

class EqualizedLayer(nn.Module):
    """
    Wrapper to apply He's initialization to weights at runtime
    because ????
    """
    
    def __init__(self, module, equalized=True):
        super(EqualizedLayer, self).__init__()

        self.module = module
        self.equalized = equalized
        
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.he_cons = self._get_he_constant(self.module)

    def forward(self, x):
        
        x = self.module(x)
        if self.equalized:
            x *= self.he_cons
        return x
            
    @staticmethod
    def _get_he_constant(x):
        size = x.weight.size()
        fan_in = np.prod(size[1:])

        return math.sqrt(2.0 / fan_in)


class EqualizedLinear(EqualizedLayer):
    """
    Linear layer with He's initialization
    """
    
    def __init__(self, dim_in, dim_out, equalized = True):
        super(EqualizedLinear, self).__init__(
            nn.Linear(dim_in, dim_out), equalized
        )
        

class EqualizedConv2d(EqualizedLayer):
    """
    Conv2d layer with He's initialization
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, equalized = True):
        super(EqualizedConv2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), equalized
        )    

class MiniBatchSD(nn.Module):
    """
    Take tensor of size BxCxHxW and produces a tensor of size B x C+1 x H x W
    where the additional dimension is a constant tensor with value equal to the
    the average of all standard deviations of each channel and each location
    in the minibatch
    """

    def __init__(self):
        super(MiniBatchSD, self).__init__()

    def forward(self, x):

        sd = torch.std(x, dim = 0)
        mean_sd = torch.mean(sd)
        sd_expanded = mean_sd.expand(x.size(0), 1, x.size(2), x.size(3))

        return torch.cat((x, sd_expanded), dim = 1)