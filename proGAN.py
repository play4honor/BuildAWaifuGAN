import torch
import torch.nn as nn
import torch.nn.functional as F
from specialLayers import EqualizedLinear, EqualizedConv2d, MiniBatchSD, WeirdoNorm, BilinearScaler

class Interpolator2x(nn.Module):
    def __init__(self):

        super(Interpolator2x, self).__init__()
        self.scale = 2
        self.mode = 'bilinear'
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)

def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

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

        self.leakiness = leakiness

        # Initialize upsampler
        self.upsampler = Upscale2d
        # self.upsampler = Interpolator2x()

        # Latent to 4x4
        self.latentNorm = WeirdoNorm()
        self.fromLatent = EqualizedLinear(latentDim, 4*4*firstLayerDepth)

        self.layers = nn.ModuleList()
        self.scales = [firstLayerDepth]
        
        # Initial layer
        self.layers.append(nn.ModuleList())
        self.layers[0].append(EqualizedConv2d(firstLayerDepth, firstLayerDepth, 3, padding=1))
        self.layers[0].append(nn.LeakyReLU(self.leakiness))
        self.layers[0].append(nn.BatchNorm2d(firstLayerDepth))
        self.layers[0].append(EqualizedConv2d(firstLayerDepth, firstLayerDepth, 3, padding=1))
        self.layers[0].append(nn.LeakyReLU(self.leakiness))
        self.layers[0].append(nn.BatchNorm2d(firstLayerDepth))

        # Last Convolution -> RGB
        self.toRGB = nn.ModuleList()
        self.toRGB.append(nn.ModuleList())
        self.toRGB[0].append(EqualizedConv2d(firstLayerDepth, self.outputDepth, 1))
        
    def set_alpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha
        

    def add_layer(self, newLayerDepth):

        self.layers.append(nn.ModuleList())
        self.layers[-1].append(EqualizedConv2d(self.scales[-1], newLayerDepth, 3, padding=1))
        self.layers[-1].append(nn.LeakyReLU(self.leakiness))
        self.layers[-1].append(WeirdoNorm())
        self.layers[-1].append(EqualizedConv2d(newLayerDepth, newLayerDepth, 3, padding=1))
        self.layers[-1].append(nn.LeakyReLU(self.leakiness))
        self.layers[-1].append(WeirdoNorm())
        
        self.toRGB.append(nn.ModuleList())
        self.toRGB[-1].append(EqualizedConv2d(newLayerDepth, self.outputDepth, 1))
        self.scales.append(newLayerDepth)

        return {
            "params": (
                [p for p in self.layers[-1].parameters() if p.requires_grad] 
                + [p for p in self.toRGB[-1].parameters() if p.requires_grad]
            )
        }

    
    def forward(self, x):
        
        # Transform latent vector into 4x4
        x = self.latentNorm(x)
        x = F.leaky_relu(self.fromLatent(x), self.leakiness)
        x = self.latentNorm(x)
        
        # batch x channels x 4 x 4
        x = x.view(x.shape[0], -1, 4, 4)

        for i, layer in enumerate(self.layers):

            if i > 0:
                x = self.upsampler(x)

            for m in layer:
                x = m(x)

            if self.alpha > 0 and i == (len(self.layers) - 2):
                y = x.clone()
                for m in self.toRGB[-2]:
                    y = m(y)
                y = self.upsampler(y)

        for m in self.toRGB[-1]:
            x = m(x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        return torch.sigmoid(x)

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

        self.leakiness = leakiness

        # Initialize downsampler
        self.downsampler = BilinearScaler(0.5)

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

        
    def set_alpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha
        

    def add_layer(self, newLayerDepth):

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
        if self.alpha > 0 and len(self.layers) > 1:
            y = self.downsampler(x)
            # Apply the fromRGB conv layer and the leaky relu.
            for m in self.fromRGB[-2]:
                y = m(y)

        # Transform RBG image to latest layer channels
        for m in self.fromRGB[-1]:
            x = m(x)
        
        # x is currently 2x the size of y

        for i, layer in enumerate(reversed(self.layers)):
            
            for m in layer: 
                x = m(x)

            if i < (len(self.layers) - 1):
                x = self.downsampler(x)

            if self.alpha > 0 and i == 0:
                x = self.alpha * y + (1.0-self.alpha) * x

        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.outputLayer(x), self.leakiness)
        x = self.finalLayer(x)

        return x, torch.Tensor([0])

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ProGANScheduler():

    """Controls scheduling for progressive GAN alpha (layer mixing) and scaling
    up, in terms of epochs."""
    def __init__(self, epochs_per_step, num_batches, scale_steps: int = 4):
        assert epochs_per_step > 1
        self.epochs_per_step = epochs_per_step
        self.num_batches = num_batches
        self.scale_steps = scale_steps

    def get_alpha(self, epoch, batch):
        if int(epoch / self.epochs_per_step) % 2 == 0:
            return 0.0
        else:
            return 1 - ((1 + batch + (epoch % self.epochs_per_step) * self.num_batches) / (1 + self.epochs_per_step * self.num_batches))
    
    def decide_scale(self, epoch):
        return (epoch % (2 * self.epochs_per_step) == self.epochs_per_step) & (epoch != 0)

    def get_max_epochs(self):
        return 2 * self.epochs_per_step * self.scale_steps + self.epochs_per_step

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

    test_gen.add_layer(32)
    test_dis.add_layer(64)

    print("Added Layers:")

    out = test_gen(x)
    print(out.shape)

    p = test_dis(out)
    print(p.shape)

    loss = test_dis(out)
    test_gen.set_alpha(0.5)
    test_dis.set_alpha(0.5)

    print("With Alpha:")

    out = test_gen(x)
    print(out.shape)

    p = test_dis(out)
    print(p.shape)
