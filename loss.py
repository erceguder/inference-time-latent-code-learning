import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# taken from original stylegan2 repository
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

# taken from original stylegan2 repository
def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def subnetworks(vgg, max_layers=5):
    subnetworks = []
    layers = []
    i = 0

    # get feature maps from layers until max_layers of Conv2D layers
    for layer in vgg:
        if isinstance(layer, torch.nn.ReLU):
            layer = torch.nn.ReLU(inplace=False)

        if i < max_layers:
            layers.append(layer)
        else: break

        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            subnetworks.append(torch.nn.Sequential(*layers))

    return subnetworks

def gramian_matrix(subnetworks, img, max_layers=5):
    assert img.ndim == 4 and img.shape[0] == 1, "Provide one image only."

    layer_acts = [net(img) for net in subnetworks]

    # convert the feature maps of each layer to gramian matrices
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

    # In the style transfer paper above, the squared error term between grams is divided by 4 * R_l^2 * C_l^2 (Eq. 4),
    # similar to the code provided by the authors, **contrary to what's stated in the paper.**
    gramians = []

    for map_ in layer_acts:
        # reshape to (C, H*W)
        map_ = map_.reshape(map_.shape[1], -1)

        # R_l & C_l in the paper
        R_l, C_l = map_.shape

        gramian = torch.matmul(map_, map_.transpose(0, 1))
        gramian = gramian.div(R_l * C_l)

        gramians.append(gramian)

    return gramians

def style_loss(subnetworks, real_img, fake_img):
    real_img = torch.unsqueeze(real_img, axis=0) if real_img.ndim==3 else real_img
    fake_img = torch.unsqueeze(fake_img, axis=0) if fake_img.ndim==3 else fake_img

    real_grams = gramian_matrix(subnetworks, real_img)
    fake_grams = gramian_matrix(subnetworks, fake_img)

    loss = 0.0

    for real_map, fake_map in zip(real_grams, fake_grams):
        #loss += torch.sum((real_map - fake_map) ** 2)
        loss += F.mse_loss(real_map, fake_map)

    return loss