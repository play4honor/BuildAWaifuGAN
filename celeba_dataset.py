import torch
import torchvision
import os

from torchvision.datasets import CelebA
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode

from specialLayers import BilinearScaler

from typing import Tuple, Any

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
        self.doubleDownsampler = None
        self.alpha = 0
        self.data_size = data_size
        self.upsampler = BilinearScaler(2)

        self.readMode = ImageReadMode.GRAY if use_greyscale else ImageReadMode.UNCHANGED

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        raw_image = torchvision.io.read_image(
                os.path.join(self.root,
                             self.base_folder,
                             "img_align_celeba",
                             self.filename[index]),
                self.readMode    
            )

        if self.downsampler is not None:

            p = self.downsampler(raw_image.type(torch.FloatTensor)) / 255.0

            if self.alpha > 0:

                y = self.doubleDownsampler(raw_image.type(torch.FloatTensor)) / 255.0
                y = self.upsampler(y)

                p = y * self.alpha + p * (1.0 - self.alpha)

            return p

    def __len__(self):
        if self.data_size is None:
            return super(CelebDataset, self).__len__()
        else:
            return self.data_size

    def get_scale(self):
        """
        Get the scale of the images
        """
        return self.scale

    def set_scale(self, scale):
        """
        Set the scale of the images
        """
        self.scale = scale
        self.doubleDownsampler = self.downsampler
        self.downsampler = Resize((self.scale, self.scale))

    def set_alpha(self, alpha):
        """
        Set the alpha value for mixing.
        """
        self.alpha = alpha

if __name__=="__main__":
    faceDS = CelebDataset(data_size=50, use_greyscale=True, root='./', split='valid', download=True)
    faceDS.set_scale(8)

    print(next(iter(faceDS)))
