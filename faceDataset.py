import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os


class FaceDataset(Dataset):
    
    IMG_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __init__(self, path, recursive=True, ext=".jpg"):
        """
        This is a docstring
        """
        super(FaceDataset, self).__init__()

        self.ext = ext
        self.path = path
        self.downsampler = None

        if self.ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image extension")

        self.fileList = self._parse_path(self.path)
        
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
        p = torchvision.io.read_image(p)
        if self.downsampler is not None:
            return self.downsampler(p.type(torch.FloatTensor))
        else:
            return p.type(torch.FloatTensor)

    def __len__(self):

        return len(self.fileList)

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

    a = FaceDataset("./img/input")

    faceLoader = DataLoader(a, batch_size=32, shuffle=True)

    b = next(iter(faceLoader))

    print(b)
    print(b.shape)
    print(b.dtype)

    faceLoader.dataset.setScale(64)

    c = next(iter(faceLoader))

    print(c)
    print(c.shape)
