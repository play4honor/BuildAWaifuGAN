from baseGan import WassersteinLoss, BaseGAN
from proGAN import ProDis, ProGen, ProGANScheduler
from torchvision.datasets import CelebA
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize

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
    def __init__(self, *args, **kwargs):
        super(CelebDataset, self).__init__(*args, **kwargs)

        self.downsampler = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.downsampler is not None:
            transform = Compose([
                CenterCrop(128),
                self.downsampler,
                ToTensor()
            ])
        else:
            transform = Compose([
                CenterCrop(128),
                ToTensor()
            ])

        X = transform(X)

        return X

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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

use_greyscale = False
channels = 1 if use_greyscale else 3

faceDS = CelebDataset('./', split='valid', download=True)

trainLoader = DataLoader(faceDS, batch_size=32, shuffle=True)

# Set up GAN

gan = BaseGAN(128, 0.01, device)

gan.setLoss(WassersteinLoss())

generator = ProGen(latentDim=128, firstLayerDepth=128, outputDepth=channels)
genOptim = AdamW(filter(lambda p: p.requires_grad, generator.parameters()))

gan.setGen(generator, genOptim)

discriminator = ProDis(firstLayerDepth=128, inputDepth=channels)
disOptim = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()))

gan.setDis(discriminator, disOptim)

scheduler = ProGANScheduler(10, len(trainLoader), scale_steps=4)
num_epochs = scheduler.get_max_epochs()

# Training

if __name__ == "__main__":

    writer = SummaryWriter()

    # Set the real image data scale
    trainLoader.dataset.setScale(4)

    j = 0

    for epoch in range(num_epochs):
        print(f"starting epoch {epoch}...")

        if scheduler.decide_scale(epoch):

            print(f"Increasing Scale to: {trainLoader.dataset.getScale()*2}")
            curr_scale = trainLoader.dataset.getScale()
            trainLoader.dataset.setScale(curr_scale*2)
            # TKTK: Add a method to base_gan to do this whole operation
            gan.generator.addLayer(128)
            gan.discriminator.addLayer(128)
            gan.generator.to(device)
            gan.discriminator.to(device)
            
        tbStep = 0

        for i, data in enumerate(trainLoader):
            x = data.to(device)

            alpha = scheduler.get_alpha(epoch, i)
            gan.generator.setAlpha(alpha)
            gan.discriminator.setAlpha(alpha)

            stepLossDis = gan.trainDis(x)

            stepLossGen, outputs = gan.trainGen(32)

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
