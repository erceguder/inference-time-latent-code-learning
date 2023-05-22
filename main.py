from model import Generator, Discriminator
from latent_learner import LatentLearner
from dataset import Dataset
from tqdm import tqdm
import loss

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import gc

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    torch.manual_seed(796)
    np.random.seed(796)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device) 
    discriminator = Discriminator(size=256).to(device)
    latent_learner = LatentLearner().to(device)

    vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()

    requires_grad(generator, False)
    requires_grad(vgg, False)

    subnetworks = loss.subnetworks(vgg, max_layers=5)

    del vgg
    gc.collect()
    torch.cuda.empty_cache()

    ckpt = torch.load("550000.pt")

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    discriminator.load_state_dict(ckpt["d"])

    # optimizers
    disc_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr = 5e-4,
        betas = (0.0, 0.99)
    )
    latent_learner_opt = torch.optim.Adam(
        latent_learner.parameters(),
        lr = 5e-4,
        betas = (0.0, 0.99)
    )
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = Dataset(path="./babies", transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    x = torch.randn(10, 1024).to(device)

    # 150 iterations
    for iter_ in tqdm(range(50)):        
        for imgs in loader:
            imgs = imgs.to(device)

            x_idx = np.random.choice(len(x), size=imgs.shape[0])

            ##### Adversarial Loss ##### 
            # first forward pass
            w = latent_learner(x[x_idx])
            samples, _ = generator([w], input_is_latent=True)

            fake_scores = discriminator(samples)
            real_scores = discriminator(imgs)

            d_loss = loss.d_logistic_loss(real_scores, fake_scores)

            # optimization step on discriminator
            d_loss.backward()
            disc_opt.step()
            disc_opt.zero_grad()

            # second forward pass
            w = latent_learner(x[x_idx])
            samples, _ = generator([w], input_is_latent=True)

            fake_scores = discriminator(samples)

            g_loss = 5 * loss.g_nonsaturating_loss(fake_scores)

            # optimization step on latent learner
            g_loss.backward()
            latent_learner_opt.step()
            latent_learner_opt.zero_grad()

            ##### Style Loss #####
            w = latent_learner(x[x_idx])
            samples, _ = generator([w], input_is_latent=True)

            vgg_loss = 0.0

            for i, img in enumerate(imgs):
                vgg_loss += 50 * loss.style_loss(subnetworks, img, samples[i])

            #print(f"d_loss: {d_loss}, g_loss: {g_loss}")#, style loss: {vgg_loss}")

            vgg_loss.backward()

        if (iter_+1) % 50 == 0:
            with torch.no_grad():
                w = latent_learner(x)
                samples, _ = generator([w])

                torchvision.utils.save_image(
                    samples.detach().clamp_(-1, 1),
                    f"samples_{iter_+1}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )