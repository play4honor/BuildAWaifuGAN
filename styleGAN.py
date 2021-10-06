import torch
import torch.nn as nn
import torch.nn.functional as F
from specialLayers import (
    EqualizedConv2d,
    LatentMapping,
    AdaIN,
    BilinearScaler
)
from dataclasses import dataclass

@dataclass
class StyleConfig:
    latent_noise_size: int = 64
    latent_mapping_size: int = 128
    latent_mapping_layers: int = 8
    output_depth: int = 3
    leakiness: float = 0.2
    channel_depth: int = 128

class StyleGen(nn.Module):

    def __init__(
        self,
        config: StyleConfig,
    ):

        super(StyleGen, self).__init__()
        
        self.config = config

        # Initialize the blending parameter
        self.alpha = 0

        self.leakiness = self.config.leakiness

        # Initialize upsampler
        self.upsampler = BilinearScaler(factor=2)

        # Intialize latent mapping
        self.latent_mapping = LatentMapping(
            self.config.latent_noise_size,
            self.config.latent_mapping_layers,
            self.config.latent_mapping_size,
            self.config.leakiness,
        )

        # Initialize Synthesis Network
        self.synthesis_layers = nn.ModuleList()

        self.learned_input = nn.Parameter(torch.ones(1, self.config.channel_depth, 4, 4))
        
        # Initial layer
        self.synthesis_layers.append(nn.ModuleList())
        self.synthesis_layers[0].append(EqualizedConv2d(self.config.channel_depth, self.config.channel_depth, 3, padding=1))
        self.synthesis_layers[0].append(nn.LeakyReLU(self.config.leakiness))
        self.synthesis_layers[0].append(AdaIN(self.config.latent_mapping_size, self.config.channel_depth))
        self.synthesis_layers[0].append(EqualizedConv2d(self.config.channel_depth, self.config.channel_depth, 3, padding=1))
        self.synthesis_layers[0].append(nn.LeakyReLU(self.config.leakiness))
        self.synthesis_layers[0].append(AdaIN(self.config.latent_mapping_size, self.config.channel_depth))

        # Last Convolution -> RGB
        self.toRGB = nn.ModuleList()
        self.toRGB.append(nn.ModuleList())
        self.toRGB[0].append(EqualizedConv2d(self.config.channel_depth, self.config.output_depth, 1))
        
    def set_alpha(self, alpha: float):

        assert alpha < 1 and alpha >= 0
        self.alpha = alpha

    def add_layer(self):

        self.synthesis_layers.append(nn.ModuleList())
        self.synthesis_layers[-1].append(EqualizedConv2d(self.config.channel_depth, self.config.channel_depth, 3, padding=1))
        self.synthesis_layers[-1].append(nn.LeakyReLU(self.config.leakiness))
        self.synthesis_layers[-1].append(AdaIN(self.config.latent_mapping_size, self.config.channel_depth))
        self.synthesis_layers[-1].append(EqualizedConv2d(self.config.channel_depth, self.config.channel_depth, 3, padding=1))
        self.synthesis_layers[-1].append(nn.LeakyReLU(self.config.leakiness))
        self.synthesis_layers[-1].append(AdaIN(self.config.latent_mapping_size, self.config.channel_depth))

        self.toRGB.append(nn.ModuleList())
        self.toRGB[-1].append(EqualizedConv2d(self.config.channel_depth, self.config.output_depth, 1))

        return {
            "params": (
                [p for p in self.synthesis_layers[-1].parameters() if p.requires_grad] 
                + [p for p in self.toRGB[-1].parameters() if p.requires_grad]
            )
        }

    def get_params(self, lr):
        
        mapping_params = [p for p in self.latent_mapping.parameters() if p.requires_grad]
        synthesis_params = (
            [p for p in self.synthesis_layers.parameters() if p.requires_grad]
            + [p for p in self.toRGB.parameters() if p.requires_grad]
            + [self.learned_input]
        )

        return [
            {"params": synthesis_params, "lr": lr},
            {"params": mapping_params, "lr": lr * 0.01},
        ]


    def forward(self, z):

        # Construct Latent Mapping to W
        w = self.latent_mapping(z)
        x = self.learned_input
        
        for i, layer in enumerate(self.synthesis_layers):

            if i > 0:
                x = self.upsampler(x)

            for m in layer:
                x = m(w, x) if isinstance(m, AdaIN) else m(x)

            if self.alpha > 0 and i == (len(self.synthesis_layers) - 2):
                y = x.clone()
                for m in self.toRGB[-2]:
                    y = m(y)
                y = self.upsampler(y)

        for m in self.toRGB[-1]:
            x = m(x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        # Or sigmoid?
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':

    a = StyleConfig()
    print(a)
