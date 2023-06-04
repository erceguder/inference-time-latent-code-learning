from PIL import Image
import torch
import glob
import os

class Dataset:
    def __init__(self, path, device, transforms):
        files = glob.glob(os.path.join(path, "*.png"))

        self.imgs = list()

        for file in files:
            self.imgs.append(transforms(Image.open(file).convert("RGB")))

        self.imgs = torch.stack(self.imgs, dim=0).to(device)

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return len(self.imgs)