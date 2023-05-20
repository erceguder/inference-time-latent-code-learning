import torch

def gramian_matrix(vgg, img, max_layers=5):
    assert img.ndim == 4 and img.shape[0] == 1, "Provide one image only."

    feature_maps = []
    layers = []
    i = 0

    # get feature maps from layers until max_layers
    for module in vgg:
        if isinstance(module, torch.nn.Conv2d):
            i += 1

            if i <= max_layers:
                layers.append(module)
                net = torch.nn.Sequential(*layers)

                feature_maps.append(net(img))
            else:
                break

    # convert the feature maps to gramian matrices
    # https://en.wikipedia.org/wiki/Gram_matrix
    gramians = []

    for map_ in feature_maps:
        # reshape to (#feature maps, H*W)
        map_ = map_.reshape(map_.shape[1], -1)
        gramian = torch.matmul(map_, map_.transpose(0, 1))

        gramians.append(gramian)

    return gramians