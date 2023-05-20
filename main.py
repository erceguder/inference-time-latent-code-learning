from model import Generator, Discriminator
import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(size=256, style_dim=512, n_mlp=8).to(device)
    #discriminator = Discriminator(size=256).to(device)

    ckpt = torch.load("550000.pt")

    print(ckpt.keys())

    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()

    latent = torch.randn(1, 512, device=device)

    sample, _ = generator([latent])