from model import Generator, Discriminator
from latent_learner import LatentLearner
from dataset import Dataset

import torch
import torchvision

def generate(generator, output_name="output.png"):
    latent = torch.randn(1, 512, device=device)

    sample, _ = generator([latent])

    torchvision.utils.save_image(
        sample,
        output_name,
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device)
    discriminator = Discriminator(size=256).to(device)

    ckpt = torch.load("550000.pt")

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    discriminator.load_state_dict(ckpt["d"])

    latent_learner = LatentLearner().to(device)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    # generate(generator)

    dataset = Dataset(path="./babies", transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for img in loader:
        x = torch.randn(img.shape[0], 1024).to(device)
        w = latent_learner(x)

        sample, _ = generator([w])

        torchvision.utils.save_image(
            sample,
            "last.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        break