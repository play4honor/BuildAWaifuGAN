import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import ImageReadMode
import os
from specialLayers import BilinearScaler

class FaceDataset(Dataset):
    
    IMG_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __init__(self, path, greyscale=False, ext=".jpg", size=None):
        """
        This is a docstring
        """
        super(FaceDataset, self).__init__()

        self.ext = ext
        self.path = path
        self.downsampler = None
        self.doubleDownsampler = None
        self.alpha = 0
        self.upsampler = BilinearScaler(2)

        self.readMode = ImageReadMode.GRAY if greyscale else ImageReadMode.UNCHANGED

        if self.ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image extension")

        self.fileList = self._parse_path(self.path)

        # Optionally limit the size of the dataset
        if size is not None:
            self.fileList = self.fileList[:size]
        
    def _parse_path(self, path):

        fileList = []

        for r, _, f in os.walk(path):
            fullpaths = [os.path.join(r, fl) for fl in f]    
            fileList.append(fullpaths)

        flatList = [p for paths in fileList for p in paths]
        flatList = [f for f in filter(lambda x: self.ext in x[-5:], flatList)]
        
        return flatList

    def __getitem__(self, idx):
        
        p = self.fileList[idx]
        raw_image = torchvision.io.read_image(p, self.readMode)

        if self.downsampler is not None:

            p = self.downsampler(raw_image.type(torch.FloatTensor)) / 255.0

            if self.alpha > 0:

                y = self.doubleDownsampler(raw_image.type(torch.FloatTensor)) / 255.0
                y = self.upsampler(y)

                p = y * self.alpha + p * (1.0 - self.alpha)

            return p
        else:
            return p.type(torch.FloatTensor) / 255.0

    def __len__(self):

        return len(self.fileList)

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
        self.downsampler = torchvision.transforms.Resize((self.scale, self.scale))

    def set_alpha(self, alpha):
        """
        Set the alpha value for mixing.
        """
        self.alpha = alpha


if __name__ == '__main__':

    a = FaceDataset("./img/input", greyscale=False)

    faceLoader = DataLoader(a, batch_size=16, shuffle=True)

    # faceLoader.dataset.setScale(256)

    b = next(iter(faceLoader))

    print(a[0])

    #torchvision.utils.save_image(a[0], "single_image.png", normalize=True)

    print(b.shape)
    print(b.dtype)

    #torchvision.utils.save_image(b, "full_scale.png", normalize=True)

    faceLoader.dataset.set_scale(128)

    b = next(iter(faceLoader))

    grid = torchvision.utils.make_grid(b, nrow=4)
    #torchvision.utils.save_image(grid, "half_scale.png", normalize=True)

    faceLoader.dataset.set_scale(64)

    b = next(iter(faceLoader))

    grid = torchvision.utils.make_grid(b, nrow=4)
    #torchvision.utils.save_image(grid, "quarter_scale.png", normalize=True)