import torch
from PIL import Image
import numpy as np
import shutil
import os

@torch.no_grad()
def save_samples(experiment, x, generator, latent_learner, exp):
    if experiment == 0:
        shutil.rmtree("samples")
        os.makedirs("samples")

    generator.eval()
    latent_learner.eval()

    w = latent_learner(x)
    samples, _ = generator([w], input_is_latent=True)

    imgs = tensor_to_img(samples)

    for i, im in enumerate(imgs):
        im.save(f"samples/samples_{exp}_{i}.png",)

def tensor_to_img(samples):
    imgs = list()

    for sample in samples:
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

def save(exp, generator, latent_learner, noise):
    os.makedirs("ckpts", exist_ok=True)

    torch.save(noise, f"ckpts/noise_{exp}.pt")
    torch.save(latent_learner.state_dict(), f"ckpts/latent_learner_{exp}.pt")
    torch.save(generator.state_dict(), f"ckpts/generator_{exp}.pt")

# taken from https://stackoverflow.com/a/65583584
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid