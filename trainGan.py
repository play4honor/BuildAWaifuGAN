from base_gan import WassersteinLoss, ModelConfig, BaseGAN
from proGAN import ProDis, ProGen
from faceDataset import FaceDataset

import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
import torchvision

device = "cuda:0" if torch.cuda.is_available() else "cpu"
faceDS = FaceDataset("./img/input")

trainDS = Subset(faceDS, range(0, int(len(faceDS) * 0.8)))
testDS = Subset(faceDS, range(int(len(faceDS) * 0.8), len(faceDS)))

trainLoader = DataLoader(trainDS, batch_size=32, shuffle=True)
testLoader = DataLoader(testDS, batch_size=32, shuffle=True)


# Set up GAN

gan = BaseGAN(128, 0.01, device)

gan.setLoss(WassersteinLoss())

generator = ProGen(latentDim=128, firstLayerDepth=64)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()))

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=64)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()))

gan.setDis(discriminator, disOptim)

# Training

if __name__ == "__main__":

    writer = SummaryWriter()

    # Set the real image data scale
    trainLoader.dataset.dataset.setScale(4)

    for epoch in range(10):

        if epoch == 5:
            trainLoader.dataset.dataset.setScale(8)
            # TKTK: Add a method to base_gan to do this whole operation
            gan.generator.addLayer(64)
            gan.discriminator.addLayer(64)
            gan.generator.to(device)
            gan.discriminator.to(device)

        tbStep = 0

        for i, data in enumerate(trainLoader):

            x = data.to(device)

            stepLossDis = gan.trainDis(x)

            stepLossGen, outputs = gan.trainGen(16)

            if i % 10 == 9:
                j = tbStep + int(len(trainLoader) / 10) * epoch
                grid = torchvision.utils.make_grid(x, nrow=4)
                writer.add_image("input", grid, j)
                grid = torchvision.utils.make_grid(outputs, nrow=4)
                writer.add_image("output", grid, j)
                writer.add_scalar("loss_discriminator", stepLossDis, j)
                writer.add_scalar("loss_generator", stepLossGen, j)
                tbStep += 1
