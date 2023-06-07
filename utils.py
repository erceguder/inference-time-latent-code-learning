import torch
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

    imgs = tensor_to_img(samples)

    for i, im in enumerate(imgs):
        im.save(f"samples/samples_{iter_}_{i}.png",)

def tensor_to_img(samples):
    imgs = list()

    for i, sample in enumerate(samples):
        sample = sample.cpu().detach().clamp_(min=-1, max=1)
        sample = (sample + 1) * 127.5
        sample = np.array(sample, dtype=np.uint8)

        # permute to (H, W, C)
        im = Image.fromarray(np.transpose(sample, (1, 2, 0)))
        imgs.append(im)

    return imgs

def disable_grad(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

def save(generator, latent_learner, noise):
    os.makedirs("ckpts", exist_ok=True)

    torch.save(noise, "ckpts/noise.pt")
    torch.save(latent_learner.state_dict(), "ckpts/latent_learner.pt")
    torch.save(generator.state_dict(), "ckpts/generator.pt")

# taken from https://stackoverflow.com/a/65583584
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid