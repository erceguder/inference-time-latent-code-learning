from model import Generator, Discriminator
from latent_learner import LatentLearner
from loss import d_logistic_loss, g_nonsaturating_loss, style_loss
from dataset import Dataset

import torch
import torchvision

import numpy as np

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device) 
    discriminator = Discriminator(size=256).to(device)
    latent_learner = LatentLearner().to(device)

    vgg = torchvision.models.vgg19(pretrained=True).features.to(device).eval()

    requires_grad(generator, False)
    requires_grad(discriminator, True)
    requires_grad(latent_learner, True)
    requires_grad(vgg, False)

    ckpt = torch.load("550000.pt")

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    discriminator.load_state_dict(ckpt["d"])

    # optimizers
    disc_opt = torch.optim.AdamW(
        discriminator.parameters(),
        lr = 5e-4
    )
    latent_learner_opt = torch.optim.AdamW(
        latent_learner.parameters(),
        lr = 5e-4
    )

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    dataset = Dataset(path="./babies", transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    x = torch.randn(10, 1024).to(device)

    # 50 iterations
    for _ in range(50):        
        for img in loader:
            img = img.to(device)

            x_idx = np.random.choice(len(x), size=img.shape[0])
            w = latent_learner(x[x_idx])

            sample, _ = generator([w])

            fake_scores = discriminator(sample)
            real_scores = discriminator(img)

            d_loss = d_logistic_loss(real_scores, fake_scores)

            # optimization step on discriminator
            d_loss.backward(retain_graph=True)
            disc_opt.step()
            disc_opt.zero_grad()

            g_loss = g_nonsaturating_loss(fake_scores)

            # optimization step on latent learner
            g_loss.backward(retain_graph=True)
            latent_learner_opt.step()
            latent_learner_opt.zero_grad()

            #for img in img:
            vgg_loss = style_loss(vgg, img, sample)
            print(f"d_loss: {d_loss}, g_loss: {g_loss}, style loss: {vgg_loss}")

            vgg_loss.backward()