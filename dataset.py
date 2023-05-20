from PIL import Image
import glob
import os

class Dataset:
    def __init__(self, path, transforms):
        files = glob.glob(os.path.join(path, "*.png"))
        self.imgs = list()

        for file in files:
            self.imgs.append(transforms(Image.open(file)))

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return len(self.imgs)