from baseGan import WassersteinLoss, ModelConfig, BaseGAN
from proGAN import ProDis, ProGen, ProGANScheduler
from faceDataset import FaceDataset

import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms.functional as TF

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

use_greyscale = False
channels = 1 if use_greyscale else 3
data_size = 5000

LATENT_SIZE = 128
LAYER_SIZE = 128
BATCH_SIZE = 32

faceDS = FaceDataset("./img/input", greyscale=use_greyscale, size=data_size)
trainLoader = DataLoader(faceDS, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batches: {len(trainLoader)}")

# Set up GAN

gan = BaseGAN(LATENT_SIZE, 0.001, device)

gan.setLoss(WassersteinLoss())

generator = ProGen(latentDim=LATENT_SIZE, firstLayerDepth=LAYER_SIZE, outputDepth=channels)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr = 0.001)

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=LAYER_SIZE, inputDepth=channels)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr = 0.001)

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

    write_batch = True

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}...")

        if scheduler.decide_scale(epoch):

            write_batch = True

            print(f"Increasing Scale to: {trainLoader.dataset.getScale()*2}")
            curr_scale = trainLoader.dataset.getScale()
            trainLoader.dataset.setScale(curr_scale*2)
            # TKTK: Add a method to base_gan to do this whole operation
            new_gen_layers = gan.generator.addLayer(LAYER_SIZE)
            new_dis_layers = gan.discriminator.addLayer(LAYER_SIZE)
            gan.generator.to(device)
            gan.discriminator.to(device)

            gan.gen_optimizer.add_param_group(new_gen_layers)
            gan.dis_optimizer.add_param_group(new_dis_layers)

        tbStep = 0

        for i, data in enumerate(trainLoader):
            x = data.to(device)

            alpha = scheduler.get_alpha(epoch, i)
            trainLoader.dataset.setAlpha(alpha)
            gan.generator.setAlpha(alpha)
            gan.discriminator.setAlpha(alpha)

            stepLosses = gan.trainDis(x)

            stepLossGen, outputs = gan.trainGen(BATCH_SIZE)

            #if i % 10 == 9:
            if True:

                obs = (1+j) * BATCH_SIZE    

                grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
                writer.add_image("input", grid, obs)

                grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True)
                if write_batch:

                    scale = trainLoader.dataset.getScale()
                    print([grid.shape[1] * (64 // scale), grid.shape[2] * (64 // scale)])
                    resize_grid = TF.resize(
                        grid, 
                        [grid.shape[1] * (64 // scale), grid.shape[2] * (64 // scale)],
                        TF.InterpolationMode.NEAREST
                    )
                    torchvision.utils.save_image(
                        resize_grid,
                        f"post_scale_output_{trainLoader.dataset.getScale()}.png",
                        normalize=True
                    )

                write_batch = False

                writer.add_image("output", grid, obs)
                writer.add_scalar("loss_discriminator", stepLosses["total_loss"], obs)
                writer.add_scalar("loss_generator", stepLossGen, obs)
                #writer.add_scalar("grad_penalty", stepLosses["grad_loss"], obs)
                #writer.add_scalar("non_grad_loss", stepLosses["non_grad_loss"], obs)
                #writer.add_scalar("real_dis_loss", stepLosses["dis_real"], obs)
                #writer.add_scalar("fake_dis_loss", stepLosses["dis_fake"], obs)

                tbStep += 1
                j += 1
