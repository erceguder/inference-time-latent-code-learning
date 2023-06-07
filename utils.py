import torch
#import torchvision
from PIL import Image
import numpy as np
import os

@torch.no_grad()
def save_samples(x, generator, latent_learner, iter_):
    os.makedirs("samples", exist_ok=True)

    generator.eval()
    latent_learner.eval()

    w = latent_learner(x)
    samples, _ = generator([w], input_is_latent=True)

    for i, sample in enumerate(samples):
        sample = sample.cpu().detach().clamp_(min=-1, max=1)
        sample = (sample + 1) * 127.5
        sample = np.array(sample, dtype=np.uint8)

        im = Image.fromarray(np.transpose(sample, (1, 2, 0)))
        im.save(f"samples/samples_{iter_}_{i}.png",)

def disable_grad(model):
    for _, param in model.named_parameters():
        param.requires_grad = False