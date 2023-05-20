from model import Generator, Discriminator

import torch
import torchvision

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device)
    #discriminator = Discriminator(size=256).to(device)

    ckpt = torch.load("550000.pt")

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    latent = torch.randn(1, 512, device=device)

    sample, _ = generator([latent])

    torchvision.utils.save_image(
        sample,
        f"output4.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )