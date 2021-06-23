import torch
import torch.nn as nn

class StandInGenerator(nn.Module):

    def __init__(self, inSize: int, outSize: tuple, outChannels: int = 3):

        super(StandInGenerator, self).__init__()

        self.inSize = inSize
        self.outSize = outSize
        self.outChannels = outChannels

        self.fc1 = nn.Linear(
            self.inSize, self.outSize[0] * self.outSize[1] * self.outChannels
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):

        # batch x input size
        x = self.fc1(x)
        x = self.sigmoid(x)
        
        return torch.reshape(
            x, (-1, self.outSize[0], self.outSize[1], self.outChannels)
        )

if __name__ == '__main__':

    from PIL import Image
    import numpy as np

    net = StandInGenerator(64, (256, 256), 3)

    input = torch.randn((16, 64))

    out = net(input)

    for i in range(16):

        im = out.detach().numpy()[i, :, :, :] * 255
        im = im.astype("uint8")
        im = Image.fromarray(im, mode="RGB")

        im.save(f"img/out/test_{i}.png")
