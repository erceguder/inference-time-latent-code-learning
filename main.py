from model import Generator, Discriminator
from latent_learner import LatentLearner
from loss import d_logistic_loss, g_nonsaturating_loss
from dataset import Dataset

import torch
import torchvision

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device)
    discriminator = Discriminator(size=256).to(device)
    latent_learner = LatentLearner().to(device)

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # 50 iterations
    for _ in range(50):
        for img in loader:
            img = img.to(device)

            x = torch.randn(img.shape[0], 1024).to(device)
            w = latent_learner(x)

            sample, _ = generator([w])

            fake_scores = discriminator(sample)
            real_scores = discriminator(img)

            #d_loss = d_logistic_loss(real_scores, fake_scores)

            # optimization step on discriminator
            #d_loss.backward()
            #disc_opt.step()
            #disc_opt.zero_grad()

            #g_loss = g_nonsaturating_loss(fake_scores)

            # optimization step on latent learner