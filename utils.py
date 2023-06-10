import torch
import numpy as np
import shutil
import os
import subprocess

from prettytable import PrettyTable
from PIL import Image
from model import Generator
from latent_learner import LatentLearner

def disable_grad(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

@torch.no_grad()
def save_samples(metadata, samples_root, experiment, x, generator, latent_learner):
    path = os.path.join(samples_root, metadata["path"])

    if experiment == 0:
        os.makedirs(samples_root, exist_ok=True)
        try:
            os.makedirs(path)
        except:
            shutil.rmtree(path)
            os.makedirs(path)

    generator.eval()
    latent_learner.eval()

    w = latent_learner(x)
    samples, _ = generator([w], input_is_latent=True)

    imgs = tensor_to_img(samples)

    for i, im in enumerate(imgs):
        im.save(os.path.join(path, f"samples_{experiment}_{i}.png"))

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

def save_models(metadata, ckpts_root, experiment, generator, latent_learner, noise):
    path = os.path.join(ckpts_root, metadata["path"])

    if experiment == 0:
        os.makedirs(ckpts_root, exist_ok=True)
        try:
            os.makedirs(path)
        except:
            shutil.rmtree(path)
            os.makedirs(path)

    torch.save(generator.state_dict(), os.path.join(path, f"generator_{experiment}.pt"))
    torch.save(latent_learner.state_dict(), os.path.join(path, f"latent_learner_{experiment}.pt"))
    torch.save(noise, os.path.join(path, f"noise_{experiment}.pt"))

def load_models(metadata, ckpts_root, experiment, resolution, w_size, n_mlp, device):
    path = os.path.join(ckpts_root, metadata["path"])

    generator = Generator(size=resolution, style_dim=w_size, n_mlp=n_mlp).to(device)
    generator.load_state_dict(torch.load(os.path.join(path, f"generator_{experiment}.pt"), map_location=device))

    latent_learner = LatentLearner().to(device)
    latent_learner.load_state_dict(torch.load(os.path.join(path, f"latent_learner_{experiment}.pt"), map_location=device))

    noise = torch.load(os.path.join(path, f"noise_{experiment}.pt"), map_location=device)

    generator.eval()
    latent_learner.eval()

    return generator, latent_learner, noise

# taken from https://stackoverflow.com/a/65583584
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid

def FID(samples_root, data_root, metadata, device):
    our_fid_scores = list()
    org_fid_scores = list()

    table = PrettyTable()
    table.field_names = ["Method"] + list(metadata.keys())

    for dset in metadata.keys():
        args = ["python",  "-m", "pytorch_fid"]
        args.append(os.path.join(data_root, metadata[dset]["test_path"]))
        args.append(os.path.join(samples_root, metadata[dset]["path"]))

        if device == "cuda":
            args.extend(["--device", "cuda:0"])

        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            out = p.communicate()[0].decode("ascii").split("FID")[1]
            fid = out.lstrip(':').lstrip(' ').rstrip('\n')
        except:
            fid = float('nan')

        our_fid_scores.append(float(fid))

        ##### Original implementation scores

        args = ["python",  "-m", "pytorch_fid"]
        args.append(os.path.join(data_root, metadata[dset]["test_path"]))
        args.append(os.path.join("./genda_samples", metadata[dset]["path"]))

        if device == "cuda":
            args.extend(["--device", "cuda:0"])

        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            out = p.communicate()[0].decode("ascii").split("FID")[1]
            fid = out.lstrip(':').lstrip(' ').rstrip('\n')
        except:
            fid = float('nan')

        org_fid_scores.append(float(fid))

    table.add_row(["Ours"] + our_fid_scores, divider=True)
    table.add_row(["Original"] + org_fid_scores, divider=True)

    return table