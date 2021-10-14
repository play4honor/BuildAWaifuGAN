from baseGan import WassersteinLoss, ModelConfig, BaseGAN
from proGAN import ProDis, ProGANScheduler
from styleGAN import StyleGen, StyleConfig
from faceDataset import FaceDataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms.functional as TF

import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Model Design
USE_GREYSCALE = True
LATENT_SIZE = 128
LAYER_SIZE = 128
LATENT_MAPPING_LAYERS = 8

# Training Details
BATCH_SIZE = 32
DATA_SIZE = 5_000
LEARNING_RATE = 0.001
EPOCHS_PER_STEP = 8
SCALE_STEPS = 4
WRITE_EVERY_N = 150
OPTIMIZER = "Adam"

channels = 1 if USE_GREYSCALE else 3
optimizer = getattr(optim, OPTIMIZER)

faceDS = FaceDataset("./img/input", greyscale=USE_GREYSCALE, size=DATA_SIZE)
trainLoader = DataLoader(faceDS, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batches: {len(trainLoader)}")

# Set up GAN

gen_config = StyleConfig(
    latent_noise_size=LATENT_SIZE,
    latent_mapping_size=LAYER_SIZE,
    latent_mapping_layers=LATENT_MAPPING_LAYERS,
    output_depth=channels,
    leakiness=0.2,
    channel_depth=LAYER_SIZE
)

gan = BaseGAN(LATENT_SIZE, device)

gan.setLoss(WassersteinLoss(sigmoid=False))

generator = StyleGen(gen_config)
genOptim = optimizer(generator.get_params(LEARNING_RATE))

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=LAYER_SIZE, inputDepth=channels)
disOptim = optimizer(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=LEARNING_RATE)

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(EPOCHS_PER_STEP, len(trainLoader), scale_steps=SCALE_STEPS)
num_epochs = scheduler.get_max_epochs()

print(f"Discriminator total initial weights: {gan.discriminator.num_params()}")

# Training

if __name__ == "__main__":

    writer = SummaryWriter()

    # Set the real image data scale
    trainLoader.dataset.set_scale(4)

    j = 0

    write_batch = True

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/profiler"),
        with_stack=True
    ) as profiler:
    # with torch.autograd.detect_anomaly():

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch}...")

            if scheduler.decide_scale(epoch):

                write_batch = True

                print(f"Increasing Scale to: {trainLoader.dataset.get_scale()*2}")
                curr_scale = trainLoader.dataset.get_scale()
                trainLoader.dataset.set_scale(curr_scale*2)
                # TKTK: Add a method to base_gan to do this whole operation
                new_gen_layers = gan.generator.add_layer()
                new_dis_layers = gan.discriminator.add_layer(LAYER_SIZE)
                gan.generator.to(device)
                gan.discriminator.to(device)

                gan.gen_optimizer = optimizer(generator.get_params(LEARNING_RATE))
                gan.dis_optimizer = optimizer(filter(lambda p: p.requires_grad, gan.discriminator.parameters()), lr=LEARNING_RATE)


            for i, data in enumerate(trainLoader):
                x = data.to(device)

                alpha = scheduler.get_alpha(epoch, i)
                trainLoader.dataset.set_alpha(alpha)
                gan.generator.set_alpha(alpha)
                gan.discriminator.set_alpha(alpha)

                stepLosses = gan.trainDis(x)

                stepLossGen, outputs = gan.trainGen(BATCH_SIZE)

                profiler.step()

                if j % WRITE_EVERY_N == 0:

                    obs = (1+j) * BATCH_SIZE    

                    grid = torchvision.utils.make_grid(x, nrow=4, normalize=True, value_range=(0,1))
                    writer.add_image("input", grid, obs)

                    grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True, value_range=(0,1))
                    if write_batch:

                        scale = trainLoader.dataset.get_scale()
                        resize_grid = TF.resize(
                            grid, 
                            [grid.shape[1] * (64 // scale), grid.shape[2] * (64 // scale)],
                            TF.InterpolationMode.NEAREST
                        )
                        torchvision.utils.save_image(
                            resize_grid,
                            f"post_scale_output_{trainLoader.dataset.get_scale()}.png",
                            normalize=True
                        )

                    write_batch = False

                    writer.add_image("output", grid, obs)
                    writer.add_scalar("loss_discriminator", stepLosses["total_loss"], obs)
                    writer.add_scalar("loss_generator", stepLossGen, obs)
                    writer.add_scalar("zgrad_penalty", stepLosses["grad_loss"], obs)
                    writer.add_scalar("non_grad_loss", stepLosses["non_grad_loss"], obs)
                    #writer.add_scalar("real_dis_loss", stepLosses["dis_real"], obs)
                    #writer.add_scalar("fake_dis_loss", stepLosses["dis_fake"], obs)

                j += 1


            if not os.path.isdir("./models"):
                os.makedirs("./models")
            gan.save(f"./models/Epoch_{epoch}_model.zip")
            