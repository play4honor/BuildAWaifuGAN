import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import math

class LatentMapping(nn.Module):
    """
    Submodule that handles the various linear layers from the latent input
    to the transformed space W, for perceptual linearity.
    """

    def __init__(self, noise_size:int, n_layers:int, layer_size:int, leakiness: float=0.2):

        super(LatentMapping, self).__init__()

        self.leakiness = leakiness
        self.noise_norm = WeirdoNorm()
        self.layers = nn.ModuleList()

        self.layers.append(EqualizedLinear(noise_size, layer_size))

        for _ in range(n_layers-1):
            self.layers.append(EqualizedLinear(layer_size, layer_size)) 

    def forward(self, x):

        x = self.noise_norm(x)

        for l in self.layers:

            x = l(x)
            x = F.leaky_relu(x, self.leakiness)

        return x


class AdaIN(nn.Module):
    """
    AdaIN layer for use before each convolutional layer
    """

    def __init__(self, w_size: int, channels: int):

        super(AdaIN, self).__init__()

        self.w_size = w_size
        self.channels = channels

        self.layer = EqualizedLinear(w_size, channels * 2)
        self.instance_norm = nn.InstanceNorm2d(channels)

    def forward(self, w, x):

        y = self.layer(w)
        y = y.view(self.channels, 2)

        # N x C x H x W
        normed_x = self.instance_norm(x)

        return (normed_x * (y[:, 0] + 1)) + y[:, 1]


class EqualizedLayer(nn.Module):
    """
    Wrapper to apply He's initialization to weights at runtime
    because ????
    """
    
    def __init__(self, module, equalized=True, init_zero_bias=True):
        super(EqualizedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if init_zero_bias:
            self.module.bias.data.fill_(0)
        
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

    def __init__(self, subGroupSize=4):
        super(MiniBatchSD, self).__init__()
        self.subGroupSize = subGroupSize

    def forward(self, x):

        size = x.shape
        G = int(size[0] / self.subGroupSize)
        if size[0] % self.subGroupSize != 0:
            raise ValueError(f'Batch size must be divisible by {self.subGroupSize}')
        
        y = x.view(-1, self.subGroupSize, size[1], size[2], size[3])        # B x subgroup size x channels x h x w
        y = torch.std(y, dim = 1)                                           # B x sd channels x sd h x sd w
        y = y.view(G, -1)                                                   # n groups x sds of stuff
        y = torch.mean(y, 1).view(G, 1)                                     # n groups x 1 (mean of sds of stuff)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))  # n_groups x h*w -> n groups x 1 x 1 x h x w
        y = y.expand(G, self.subGroupSize, -1, -1, -1)                      # n_groups x subgroup size x 1 x h x w
        y = y.contiguous().view((-1, 1, size[2], size[3]))                  # B x 1 x h x w

        return torch.cat((x, y), dim = 1)

class WeirdoNorm(nn.Module):
    """
    This is the weird local response normalization 
    """
     
    def __init__(self):
         super(WeirdoNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

def scaleBilinear(x, factor):

    new_size = [int(x.shape[-2] * factor), int(x.shape[-1] * factor)]

    return TF.resize(x, size = new_size)