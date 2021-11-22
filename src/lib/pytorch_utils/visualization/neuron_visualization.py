from typing import Dict, Union, Tuple

import matplotlib.pyplot as plt

import torch
from torch import nn
from warnings import warn

from ..surgery import LayerOutputExtractor_wrapper


def neuron_visualizer(model: torch.nn.Module, layer_name: str, filter_num: int = None, save_loc: str = "", starting_image: torch.Tensor = None, dataset: str = "mnist",  verbose: bool = False):
    """
    Description: Feature Visualizer for a given neuron
    Input :
        net : Neural Network                                        (torch.nn.Module)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    device = model.parameters().__next__().device
    model = LayerOutputExtractor_wrapper(model, layer_names=[layer_name])

    if not starting_image:
        if dataset == "mnist":
            starting_image = (torch.rand(1, 1, 28, 28)/10).to(device)
        elif dataset == "cifar":
            starting_image = (torch.rand(1, 3, 32, 32)/10).to(device)
        else:
            raise NotImplementedError

    optimizer = Adam([starting_image], lr=0.1, weight_decay=1e-6)

    for i in range(1, 31):
        optimizer.zero_grad()
        _ = model(starting_image)

        layer_output = model.layer_outputs[layer_name]

        layer_shape = layer_output.shape

        if filter_num:
            loss = -layer_output[0, filter_num, layer_shape[2]//2, layer_shape[3]//2]
        else:
            loss = -layer_output[0, layer_shape[1]//2, layer_shape[2]//2, layer_shape[3]//2]
            filter_num = layer_shape[1]//2

        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            im_path = save_loc + str(layer_name) + \
                '_f_' + str(filter_num) + '_iter_' + str(i) + '.jpg'
            plt.figure()
            plt.imshow(starting_image[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(im_path)
            plt.close()
