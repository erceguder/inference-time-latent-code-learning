from PIL import Image
import torch
import glob
import os

class Dataset:
    def __init__(self, root, path, extension, transforms, n_imgs=10):
        files = glob.glob(os.path.join(root, path, f"*.{extension}"))
        self.imgs = list()

        for file in files:
            self.imgs.append(transforms(Image.open(file).convert("RGB")))

        self.imgs = torch.stack(self.imgs, dim=0)
        self.x = torch.randn(n_imgs, 14, 1024)

    def __getitem__(self, idx):
        return self.imgs[idx], self.x[idx]

    def __len__(self):
        return len(self.imgs)