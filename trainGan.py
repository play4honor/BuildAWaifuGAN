from baseGan import WassersteinLoss, ModelConfig, BaseGAN
from proGAN import ProDis, ProGen, ProGANScheduler
from faceDataset import FaceDataset

import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
import torchvision

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

use_greyscale = False
channels = 1 if use_greyscale else 3
data_size = 10000

faceDS = FaceDataset("./img/input", greyscale=use_greyscale, size=data_size)
trainLoader = DataLoader(faceDS, batch_size=32, shuffle=True)
print(f"Batches: {len(trainLoader)}")

# Set up GAN

gan = BaseGAN(128, 0.001, device)

gan.setLoss(WassersteinLoss())

generator = ProGen(latentDim=128, firstLayerDepth=128, outputDepth=channels)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()))

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=128, inputDepth=channels)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()))

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(5, len(trainLoader), scale_steps=4)
num_epochs = scheduler.get_max_epochs()

print(gan.discriminator.num_params())

# Training

if __name__ == "__main__":

    writer = SummaryWriter()

    # Set the real image data scale
    trainLoader.dataset.setScale(4)

    j = 0

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}...")

        if scheduler.decide_scale(epoch):

            print(f"Increasing Scale to: {trainLoader.dataset.getScale()*2}")
            curr_scale = trainLoader.dataset.getScale()
            trainLoader.dataset.setScale(curr_scale*2)
            # TKTK: Add a method to base_gan to do this whole operation
            new_gen_layers = gan.generator.addLayer(128)
            new_dis_layers = gan.discriminator.addLayer(128)
            gan.generator.to(device)
            gan.discriminator.to(device)

            gan.gen_optimizer.add_param_group(new_gen_layers)
            gan.dis_optimizer.add_param_group(new_dis_layers)


            
        tbStep = 0

        for i, data in enumerate(trainLoader):
            x = data.to(device)

            alpha = scheduler.get_alpha(epoch, i)
            gan.generator.setAlpha(alpha)
            gan.discriminator.setAlpha(alpha)

            stepLosses = gan.trainDis(x, alpha)

            stepLossGen, outputs = gan.trainGen(32)

            #if i % 10 == 9:
            if True:
                #j = tbStep + int(len(trainLoader) / 10) * epoch
                grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
                writer.add_image("input", grid, j)
                grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True)
                writer.add_image("output", grid, j)
                writer.add_scalar("loss_discriminator", stepLosses["total_loss"], j)
                writer.add_scalar("loss_generator", stepLossGen, j)
                writer.add_scalar("grad_penalty", stepLosses["grad_loss"], j)
                writer.add_scalar("non_grad_loss", stepLosses["non_grad_loss"], j)
                writer.add_scalar("real_dis_loss", stepLosses["dis_real"], j)
                writer.add_scalar("fake_dis_loss", stepLosses["dis_fake"], j)
                tbStep += 1
                j += 1
