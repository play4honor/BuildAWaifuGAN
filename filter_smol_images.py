import torchvision
from torchvision.io import ImageReadMode

import os
import shutil

IMG_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def parse_path(path):

        ext = ".jpg"
        fileList = []

        for r, _, f in os.walk(path):
            fullpaths = [os.path.join(r, fl) for fl in f]    
            fileList.append(fullpaths)

        flatList = [p for paths in fileList for p in paths]
        flatList = [f for f in filter(lambda x: ext in x[-5:], flatList)]
        
        return flatList

def check_size(path):

    try:
        raw_image = torchvision.io.read_image(path, ImageReadMode.UNCHANGED)

    except RuntimeError:
        return False

    return raw_image.shape[1] >= 64

if __name__ == '__main__':

    path = "./img/input2"
    out_path = "./img/input3"
    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    fileList = parse_path(path)

    filtered_file_list = filter(check_size, fileList)

    for i in filtered_file_list:

        file_name = i.split('\\')[-1]

        target_path = f"{out_path}/{file_name}"

        shutil.copyfile(i, target_path)