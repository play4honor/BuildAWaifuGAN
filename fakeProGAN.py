import torch
import torch.nn as nn
import torch.nn.functional as F
from specialLayers import EqualizedLinear, EqualizedConv2d, MiniBatchSD

class Interpolatorhalfx(nn.Module):
    def __init__(self):

        super(Interpolatorhalfx, self).__init__()
        self.scale = 0.5
        self.mode = 'bilinear'
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)

class FakeProDis(nn.Module):

    def __init__(
        self,
        firstLayerDepth: int,
        inputDepth: int=3,
        leakiness: float=0.2,
        minibatchSD: bool=True,
    ):

        super(FakeProDis, self).__init__()
        
        # Output Depth
        self.inputDepth = inputDepth

        # Initialize the blending parameter
        self.alpha = 0

        self.leakiness = leakiness

        # Initialize downsampler
        self.downsampler = Interpolatorhalfx()

        self.layers = nn.ModuleList()
        self.scales = [firstLayerDepth]
        
        # Initial layer
        self.layers.append(nn.ModuleList())
        firstLayerActualDepth = firstLayerDepth
        if minibatchSD:
            self.layers[0].append(MiniBatchSD())
            firstLayerActualDepth += 1
        self.layers[0].append(EqualizedConv2d(firstLayerActualDepth, firstLayerDepth, 3, padding=1))
        self.layers[0].append(nn.LeakyReLU(self.leakiness))
        self.layers[0].append(EqualizedConv2d(firstLayerDepth, firstLayerDepth, 3, padding=1))
        self.layers[0].append(nn.LeakyReLU(self.leakiness))

        # Output Layer
        self.outputLayer = EqualizedLinear(firstLayerDepth*4*4, firstLayerDepth)
        self.finalLayer = EqualizedLinear(firstLayerDepth, 1)

        # RGB -> layer depth
        self.fromRGB = nn.ModuleList()
        self.fromRGB.append(nn.ModuleList())
        
        self.fromRGB[0].append(EqualizedConv2d(self.inputDepth, firstLayerDepth, 1))
        self.fromRGB[0].append(nn.LeakyReLU(self.leakiness))

        
    def setAlpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha
        

    def addLayer(self, newLayerDepth):

        self.layers.append(nn.ModuleList())
        self.layers[-1].append(EqualizedConv2d(self.scales[-1], newLayerDepth, 3, padding=1))
        self.layers[-1].append(nn.LeakyReLU(self.leakiness))
        self.layers[-1].append(EqualizedConv2d(newLayerDepth, newLayerDepth, 3, padding=1))
        self.layers[-1].append(nn.LeakyReLU(self.leakiness))
        
        self.fromRGB.append(nn.ModuleList())
        self.fromRGB[-1].append(EqualizedConv2d(self.inputDepth, newLayerDepth, 1))
        self.fromRGB[-1].append(nn.LeakyReLU(self.leakiness))

        self.scales.append(newLayerDepth)

        return {
            "params": (
                [p for p in self.layers[-1].parameters() if p.requires_grad] 
                + [p for p in self.fromRGB[-1].parameters() if p.requires_grad]
            )
        }
    
    def forward(self, x):
        
        # Downscaled Image
        while x.shape[2] > 4:
            x = self.downsampler(x)

        act = [x.mean(), x.std()]

        # Transform RBG image to latest layer channels
        for m in self.fromRGB[-1]:
            x = m(x)

        for i, layer in enumerate(reversed(self.layers)):
            
            for m in layer: 
                x = m(x)

        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.outputLayer(x), self.leakiness)
        x = self.finalLayer(x)

        return x, act

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
