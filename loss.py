import torch
import torch.nn.functional as F

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
        if i < max_layers:
            layers.append(layer)
        else: break

        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            subnetworks.append(torch.nn.Sequential(*layers))

    return subnetworks

def gramian_matrix(subnetworks, img, max_layers=5):
    assert img.ndim == 4 and img.shape[0] == 1, "Provide one image only."

    feature_maps = []

    # img should be normalized with
    #    mean: 0.485, 0.456, 0.406
    #    std: 0.229, 0.224, 0.225

    for net in subnetworks:
        feature_maps.append(net(img))

    # convert the feature maps to gramian matrices
    # https://en.wikipedia.org/wiki/Gram_matrix
    gramians = []

    for map_ in feature_maps:
        # reshape to (#feature maps, H*W)
        map_ = map_.reshape(map_.shape[1], -1)
        gramian = torch.matmul(map_, map_.transpose(0, 1))

        gramians.append(gramian)

    return gramians

def style_loss(subnetworks, real_img, fake_img):
    real_img = torch.unsqueeze(real_img, axis=0) if real_img.ndim==3 else real_img
    fake_img = torch.unsqueeze(fake_img, axis=0) if fake_img.ndim==3 else fake_img

    real_grams = gramian_matrix(subnetworks, real_img)
    fake_grams = gramian_matrix(subnetworks, fake_img)

    loss = 0.0

    for real_map, fake_map in zip(real_grams, fake_grams):
        loss += F.mse_loss(real_map, fake_map)

    return loss