from baseGan import WassersteinLoss, BaseGAN
from proGAN import ProDis, ProGen, ProGANScheduler
from torchvision.datasets import CelebA
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize, Lambda
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
import torchvision

import os
from typing import Tuple, Any
import PIL

class CelebDataset(CelebA):
    """
    Download these files from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    "img_align_celeba.zip"
    "list_attr_celeba.txt"
    "identity_CelebA.txt"
    "list_bbox_celeba.txt"
    "list_landmarks_align_celeba.txt"
    "list_eval_partition.txt"
    """
    def __init__(self, data_size=None, use_greyscale=True, *args, **kwargs):
        super(CelebDataset, self).__init__(*args, **kwargs)

        self.downsampler = None
        self.alpha = 0
        self.data_size = data_size
        self.use_greyscale = use_greyscale

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(
            os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
            )

        if self.use_greyscale:
            X = PIL.ImageOps.grayscale(X)

        if self.downsampler is not None:
            transform = Compose([
                CenterCrop(128),
                self.downsampler,
                ToTensor(),
                Lambda(self._scale_mixer)
            ])
        else:
            transform = Compose([
                CenterCrop(128),
                ToTensor()
            ])

        X = transform(X)

        return X

    def __len__(self):
        if self.data_size is None:
            return super(CelebDataset, self).__len__()
        else:
            return self.data_size

    def _scale_mixer(self, p):
        if self.alpha > 0:
            s = p.shape
            y = p.reshape((s[0], s[1]//2, 2, s[2]//2, 2))
            y = torch.mean(y, dim=[2, 4], keepdim=True)
            y = torch.tile(y, (1, 1, 2, 1, 2))
            y = y.reshape(s)
            p = y * self.alpha + p * (1.0 - self.alpha)
            return p

        else: 
            return p

        

    def getScale(self):
        """
        Get the scale of the images
        """
        return self.scale

    def setScale(self, scale):
        """
        Set the scale of the images
        """
        self.scale = scale
        self.downsampler = Resize((self.scale, self.scale))

    def setAlpha(self, alpha):
        """
        Set the alpha value for mixing.
        """
        self.alpha = alpha


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Model Design
USE_GREYSCALE = True
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

faceDS = CelebDataset(data_size=DATA_SIZE, use_greyscale=USE_GREYSCALE, root='./', split='valid', download=True)
trainLoader = DataLoader(faceDS, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batches: {len(trainLoader)}")

# Set up GAN

gan = BaseGAN(LATENT_SIZE, 0.001, device)

gan.setLoss(WassersteinLoss(sigmoid=False))

generator = ProGen(latentDim=LATENT_SIZE, firstLayerDepth=LAYER_SIZE, outputDepth=channels)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr = LEARNING_RATE)

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=LAYER_SIZE, inputDepth=channels)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr = LEARNING_RATE)

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(EPOCHS_PER_STEP, len(trainLoader), scale_steps=SCALE_STEPS)
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

                grid = torchvision.utils.make_grid(x, nrow=4, normalize=True, value_range=(0,1))
                writer.add_image("input", grid, obs)

                grid = torchvision.utils.make_grid(outputs, nrow=4, normalize=True, value_range=(0,1))
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
                writer.add_scalar("zgrad_penalty", stepLosses["grad_loss"], obs)
                writer.add_scalar("non_grad_loss", stepLosses["non_grad_loss"], obs)
                #writer.add_scalar("real_dis_loss", stepLosses["dis_real"], obs)
                #writer.add_scalar("fake_dis_loss", stepLosses["dis_fake"], obs)

            j += 1
