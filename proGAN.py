import torch
import torch.nn as nn
from specialLayers import EqualizedLinear, EqualizedConv2d, MiniBatchSD

class ProGen(nn.Module):

    def __init__(
        self,
        latentDim: int,
        firstLayerDepth: int,
        outputDepth: int=3,
        leakiness: float=0.2,
    ):

        super(ProGen, self).__init__()
        
        # Output Depth
        self.outputDepth = outputDepth

        # Initialize the blending parameter
        self.alpha = 0

        self.leakyReLU = nn.LeakyReLU(leakiness)

        # Initialize upsampler
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        # Latent to 4x4
        self.fromLatent = EqualizedLinear(latentDim, 4*4*firstLayerDepth)

        self.layers = nn.ModuleList()
        self.scales = [firstLayerDepth]
        
        # Initial layer
        self.layers.append(nn.ModuleList())
        self.layers[0].append(EqualizedConv2d(firstLayerDepth, firstLayerDepth, 3, padding=1))

        # Last Convolution -> RGB
        self.toRGB = nn.ModuleList()
        self.toRGB.append(EqualizedConv2d(firstLayerDepth, self.outputDepth, 1))

        
    def setAlpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha
        

    def addLayer(self, newLayerDepth):

        self.layers.append(nn.ModuleList())
        self.layers[-1].append(EqualizedConv2d(self.scales[-1], newLayerDepth, 3, padding=1))
        self.layers[-1].append(EqualizedConv2d(newLayerDepth, newLayerDepth, 3, padding=1))
        
        self.toRGB.append(EqualizedConv2d(newLayerDepth, self.outputDepth, 1))

        self.scales.append(newLayerDepth)

    
    def forward(self, x):
        
        # Transform latent vector into 4x4
        x = self.leakyReLU(self.fromLatent(x))
        
        # batch x channels x 4 x 4
        x = x.view(x.shape[0], -1, 4, 4)

        for i, layer in enumerate(self.layers):

            if i > 0:
                x = self.upsampler(x)

            for m in layer:
                x = self.leakyReLU(m(x))

            if self.alpha > 0 and i == (len(self.layers) - 2):
                y = self.toRGB[-2](x)
                y = self.upsampler(y)

        x = self.toRGB[-1](x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProDis(nn.Module):

    def __init__(
        self,
        firstLayerDepth: int,
        inputDepth: int=3,
        leakiness: float=0.2,
        minibatchSD: bool=True,
    ):

        super(ProDis, self).__init__()
        
        # Output Depth
        self.inputDepth = inputDepth

        # Initialize the blending parameter
        self.alpha = 0

        self.leakyReLU = nn.LeakyReLU(leakiness)

        # Initialize downsampler
        self.downsampler = nn.AvgPool2d(kernel_size=2)

        self.layers = nn.ModuleList()
        self.scales = [firstLayerDepth]
        
        # Initial layer
        self.layers.append(nn.ModuleList())
        firstLayerActualDepth = firstLayerDepth
        if minibatchSD:
            self.layers[0].append(MiniBatchSD())
            firstLayerActualDepth += 1
        self.layers[0].append(EqualizedConv2d(firstLayerActualDepth, firstLayerDepth, 3, padding=1))
        
        # Output Layer
        self.outputLayer = EqualizedLinear(firstLayerDepth*4*4, firstLayerDepth)
        self.finalLayer = EqualizedLinear(firstLayerDepth, 1)

        # RGB -> layer depth
        self.fromRGB = nn.ModuleList()
        self.fromRGB.append(EqualizedConv2d(self.inputDepth, firstLayerDepth, 1))

        
    def setAlpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha
        

    def addLayer(self, newLayerDepth):

        self.layers.append(nn.ModuleList())
        self.layers[-1].append(EqualizedConv2d(newLayerDepth, self.scales[-1], 3, padding=1))

        self.fromRGB.append(EqualizedConv2d(self.inputDepth, newLayerDepth, 1))

        self.scales.append(newLayerDepth)
    
    
    def forward(self, x):
        
        # Downscaled Image
        if self.alpha > 0 and len(self.layers) > 1:
            y = self.downsampler(x)
            y = self.leakyReLU(self.fromRGB[-2](y))

        # Transform RBG image to latest layer channels
        x = self.leakyReLU(self.fromRGB[-1](x))
        
        # x is currently 2x the size of y

        for i, layer in enumerate(reversed(self.layers)):
            
            for m in layer: 
                x = self.leakyReLU(m(x))

            if i < (len(self.layers) - 1):
                x = self.downsampler(x)

            if self.alpha > 0 and i == 0:
                x = self.alpha * y + (1.0-self.alpha) * x

        x = x.view(x.shape[0], -1)
        x = self.leakyReLU(self.outputLayer(x))
        x = self.finalLayer(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':


    test_gen = ProGen(
        latentDim=128,
        firstLayerDepth=64,
    )

    test_dis = ProDis(
        firstLayerDepth=32
    )

    x = torch.randn([16, 128])

    print("Base Layers:")

    out = test_gen(x)
    print(out.shape)

    loss = test_dis(out)

    p = test_dis(out)
    print(p.shape)

    test_gen.addLayer(32)
    test_dis.addLayer(64)

    print("Added Layers:")

    out = test_gen(x)
    print(out.shape)

    p = test_dis(out)
    print(p.shape)

    loss = test_dis(out)
    test_gen.setAlpha(0.5)
    test_dis.setAlpha(0.5)

    print("With Alpha:")

    out = test_gen(x)
    print(out.shape)

    p = test_dis(out)
    print(p.shape)
