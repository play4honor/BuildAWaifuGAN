import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import ImageReadMode
import os


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
        p = torchvision.io.read_image(p, self.readMode)
        if self.downsampler is not None:
            return self.downsampler(p.type(torch.FloatTensor)) / 255.0
        else:
            return p.type(torch.FloatTensor) / 255.0

    def __len__(self):

        return len(self.fileList)

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
        self.downsampler = torchvision.transforms.Resize((self.scale, self.scale))

    def view_image(self, idx):
        """
        View the image at the given index
        """
        pass

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

    faceLoader.dataset.setScale(128)

    b = next(iter(faceLoader))

    grid = torchvision.utils.make_grid(b, nrow=4)
    #torchvision.utils.save_image(grid, "half_scale.png", normalize=True)

    faceLoader.dataset.setScale(64)

    b = next(iter(faceLoader))

    grid = torchvision.utils.make_grid(b, nrow=4)
    #torchvision.utils.save_image(grid, "quarter_scale.png", normalize=True)