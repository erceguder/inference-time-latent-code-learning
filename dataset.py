from PIL import Image
import torch
import glob
import os

class Dataset:
    def __init__(self, path, extension, transforms):
        files = glob.glob(os.path.join(path, f"*.{extension}"))
        self.imgs = list()

        for file in files:
            self.imgs.append(transforms(Image.open(file).convert("RGB")))

        self.imgs = torch.stack(self.imgs, dim=0)
        self.x = torch.randn(10, 14, 1024)

    def __getitem__(self, idx):
        return self.imgs[idx], self.x[idx]

    def __len__(self):
        return len(self.imgs)