from baseGan import WassersteinLoss, ModelConfig, BaseGAN, ProGANScheduler
from proGAN import ProDis, ProGen
from faceDataset import FaceDataset

import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
import torchvision

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
faceDS = FaceDataset("./sample_data")

trainLoader = DataLoader(faceDS, batch_size=32, shuffle=True)

# Set up GAN

gan = BaseGAN(128, 0.01, device)

gan.setLoss(WassersteinLoss())

generator = ProGen(latentDim=128, firstLayerDepth=64)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()))

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=64)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()))

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(2, len(trainLoader))

# Training

if __name__ == "__main__":

    writer = SummaryWriter()

    # Set the real image data scale
    trainLoader.dataset.setScale(4)

    j = 0

    for epoch in range(10):
        print(f"starting epoch {epoch}...")

        if scheduler.decide_scale(epoch):

            print(f"Increasing Scale to: {trainLoader.dataset.getScale()*2}")
            curr_scale = trainLoader.dataset.getScale()
            trainLoader.dataset.setScale(curr_scale*2)
            # TKTK: Add a method to base_gan to do this whole operation
            gan.generator.addLayer(64)
            gan.discriminator.addLayer(64)
            gan.generator.to(device)
            gan.discriminator.to(device)
            
        tbStep = 0

        for i, data in enumerate(trainLoader):
            print(f"starting batch {i} of epoch {epoch}...")
            x = data.to(device)

            alpha = scheduler.get_alpha(epoch, i)
            print(f"alpha is {alpha}")
            gan.generator.setAlpha(alpha)
            gan.discriminator.setAlpha(alpha)

            stepLossDis = gan.trainDis(x)

            stepLossGen, outputs = gan.trainGen(16)

            #if i % 10 == 9:
            if True:
                #j = tbStep + int(len(trainLoader) / 10) * epoch
                grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
                writer.add_image("input", grid, j)
                grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True)
                writer.add_image("output", grid, j)
                writer.add_scalar("loss_discriminator", stepLossDis, j)
                writer.add_scalar("loss_generator", stepLossGen, j)
                tbStep += 1
                j += 1
