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
print(f"Using {device}")

# Model Design
USE_GREYSCALE = False
LATENT_SIZE = 192
LAYER_SIZE = 192

# Training Details
BATCH_SIZE = 32
DATA_SIZE = 15000
LEARNING_RATE = 0.001
EPOCHS_PER_STEP = 8
SCALE_STEPS = 4
WRITE_EVERY_N = 50

channels = 1 if USE_GREYSCALE else 3

faceDS = FaceDataset("./img/input", greyscale=USE_GREYSCALE, size=DATA_SIZE)
trainLoader = DataLoader(faceDS, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batches: {len(trainLoader)}")

# Set up GAN

gan = BaseGAN(LATENT_SIZE, 0.001, device)

gan.setLoss(WassersteinLoss(sigmoid=False))

generator = ProGen(latentDim=LATENT_SIZE, firstLayerDepth=LAYER_SIZE, outputDepth=channels)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=LEARNING_RATE)

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=LAYER_SIZE, inputDepth=channels)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=LEARNING_RATE)

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(EPOCHS_PER_STEP, len(trainLoader), scale_steps=SCALE_STEPS)
num_epochs = scheduler.get_max_epochs()

print(f"Discriminator total initial weights: {gan.discriminator.num_params()}")

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


        for i, data in enumerate(trainLoader):
            x = data.to(device)

            alpha = scheduler.get_alpha(epoch, i)
            trainLoader.dataset.setAlpha(alpha)
            gan.generator.setAlpha(alpha)
            gan.discriminator.setAlpha(alpha)

            stepLosses = gan.trainDis(x)

            stepLossGen, outputs = gan.trainGen(BATCH_SIZE)

            if j % WRITE_EVERY_N == 0:

                obs = (1+j) * BATCH_SIZE    

                grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
                writer.add_image("input", grid, obs)

                grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True)
                if write_batch:

                    scale = trainLoader.dataset.getScale()
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
                writer.add_scalar("grad_penalty", stepLosses["grad_loss"], obs)
                writer.add_scalar("non_grad_loss", stepLosses["non_grad_loss"], obs)
                #writer.add_scalar("real_dis_loss", stepLosses["dis_real"], obs)
                #writer.add_scalar("fake_dis_loss", stepLosses["dis_fake"], obs)

            j += 1
