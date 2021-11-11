from typing import Dict, Union, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import hoyer


__all__ = ["hoyer_loss", "l1_loss", "l2_loss"]


def hoyer_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None, epsilon: float = 0.000000000001) -> torch.float:
    """
    Hoyer square loss: https://arxiv.org/pdf/1908.09979.pdf

    Takes a dictionary of tensors

    returns (|feature|_1)^2/((|feature|_2)^2 + epsilon)
    """
    loss = 0
    for feature in features:
        loss += torch.mean(hoyer(x=features[feature], dim=dim, epsilon=epsilon)**2)
    return loss


def l1_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None) -> torch.float:
    """
    L1 loss: L1 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_1
    """
    loss = 0
    for feature in features:
        loss += torch.mean(torch.sum(torch.abs(features[feature]), dim=dim))
    return loss


def l2_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None) -> torch.float:
    """
    L2 loss: L2 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_2
    """
    loss = 0
    for feature in features:
        loss += torch.mean(torch.sqrt(torch.sum(features[feature]**2, dim=dim)))
    return loss


def saliency_K(features: Dict[str, torch.Tensor], K: int, saliency_lambda: float = 1.0, dim: Union[int, Tuple[int]] = None, **kwargs):

    # sort in channel dimension; each patch's outputs should be sparse
    loss = 0
    for feature in features:
        sorted = torch.sort(features[feature].abs(), dim=1, descending=True)[0]
        top_K_avg = sorted[:, :K].mean(dim=1)
        bottom_avg = sorted[:, K:].mean(dim=1)
        loss += top_K_avg-saliency_lambda*bottom_avg
    return loss
