import datetime
import datetime
import os
from math import ceil
from math import sqrt
from pathlib import Path

import torch
from torchvision.utils import make_grid

from activation_maximization import maximize_activation, init_input
from model.c3d import c3d
from util.save_helper import save_tensor_as_img, save_tensor_list_as_gif, save_tensor_list_as_avi

# General overview of how activation maximization works:
# Our goal is, given a neural network,
# to inspect what individual neurons, or rather whole groups of neurons
# are most 'excited' about. Practically this means finding the input
# that yields the highest activation value for single neuron
# or for a particular featuremap. This can show us what types of filters
# the network has learned. To visualize this exact scenario,
# activation maximization starts with an already trained network, to which a random
# input is supplied. Then we choose a particular neuron or featuremap (which is averaged
# to obtain a scalar). We set the gradient on that particular neuron to its activation
# and backpropagate it back to the input image. Regularization is applied that (hopefully)
# forces the image to look like a natural image and not an adversarial example.


def get_c3d_model():
    model = c3d()
    model.load_state_dict(torch.load('model/c3d.pickle'))  # This is Sports-1M pretrained
    return model


if __name__ == '__main__':
    # Srsly: figure out the mean and std of the data your model was trained on, otherwise it'll be shit and diverge
    #        pretty quickly to a single uniform color because of the re-normalization that I am applying in AM.
    num_frames = 16
    model = get_c3d_model()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    # ----------- parameters  -----------
    target_class = 0  # Not needed when maximizing individual layers
    # TRIPPY CONFIG:
    # blur_rad = 2 # 4
    # wd = 0.00001
    # kappa = 0.3
    # gamma = 0.00001 #0.0005
    # lr = 3  #0.1
    # iterations = 80
    # img_height = 112
    # img_width = 112

    # Not super shitty config
    blur_rad = 0.3 # 4
    wd = 0.00001
    kappa = 0.01
    gamma = 0.001
    lr = 0.1
    iterations = 70
    img_height = 112
    img_width = 112

    path_prefix = f'runs/'
    Path(path_prefix).mkdir(parents=True, exist_ok=True)
    now = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

    outdir = f'{path_prefix}/{model.name} blur {blur_rad:.2f} | kappa {kappa:.2e} | gamma {gamma:.2e} | lr {lr} | iter {iterations} | {now}'
    os.mkdir(outdir)
    init = init_input(num_frames, init_blur=0.1, mean=mean, std=std, img_height=img_height, img_width=img_width)

    # Implement center biasing
    # https://arxiv.org/pdf/1602.03616.pdf
    for layer in [10]:  # <--- this is the list of layers you'll inspect. make sure to select Conv3D layers [0, 4, 7 , 10, etc... ]
        created_images = init
        channels = list(model.children())[layer].out_channels

        all_frame1 = []
        all_frame2 = []

        os.mkdir(f'{outdir}/{layer}/')
        for i in range(channels):
            print(f'[{i}/{channels}]')
            # do Activ Maxim
            created_images = maximize_activation(model, target_class, created_images, iterations, blur_rad, wd, lr, kappa, gamma, mean, std, i, layer)
            created_tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in created_images]  # [ (img), (img), ...]  img: ch,w,h
            save_tensor_list_as_gif(created_tensors, path=f'{outdir}/{layer}/{i}.gif', duration=150)
            all_frame1.append(created_tensors[0])
            all_frame2.append(created_tensors[1])

        all_frame1_grid = make_grid(all_frame1, nrow=ceil(sqrt(channels)))
        all_frame2_grid = make_grid(all_frame2, nrow=ceil(sqrt(channels)))
        save_tensor_list_as_gif([all_frame1_grid, all_frame2_grid], path=f'{outdir}/{layer}_all.gif', duration=150)

    print(f'Outdir is {outdir}')
