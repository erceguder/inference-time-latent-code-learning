import torch
import torchvision
import os

@torch.no_grad()
def save_samples(x, generator, latent_learner, iter_):
    os.makedirs("samples", exist_ok=True)

    generator.eval()
    latent_learner.eval()

    w = latent_learner(x)
    samples, _ = generator([w], input_is_latent=True)

    for i, sample in enumerate(samples):
        torchvision.utils.save_image(
            sample.detach().clamp_(min=-1, max=1),
            f"samples/samples_{iter_}_{i}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

def disable_grad(model):
    for _, param in model.named_parameters():
        param.requires_grad = False