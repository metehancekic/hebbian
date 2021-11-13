"""
Neural Network models for training and testing implemented in PyTorch
"""
from typing import Dict, Iterable, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from ..custom_activations import TReLU_with_trainable_bias, TReLU
from ..custom_layers import Normalize, take_top_k, DivisiveNormalization2d, Binarization


def norm_weight(weights, lp_norm=1):
    if weights.ndim == 4:
        lp_weight = torch.norm(weights, p=lp_norm, dim=(1, 2, 3))
    elif weights.ndim == 2:
        lp_weight = torch.norm(weights, p=lp_norm, dim=1)
    else:
        raise NotImplementedError
    return lp_weight


class topk_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes=10, gamma=1., k=4):
        super().__init__()

        self.gamma = gamma
        self.k = k

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)

        self.relu = TReLU(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x):

        # out = self.norm(x)
        out = x
        self.l1 = self.conv1(out)
        # bias = self.bias_calculator(self.conv1.weight)
        # if self.training:
        #     noise = bias * 8./255 * (torch.rand_like(self.out, device=o.device)
        #                              * self.gamma * 2 - self.gamma)
        #     self.out = self.out + noise
        out = take_top_k(self.l1, self.k)
        out = self.relu(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self):
        return f"top{self.k}_LeNet"


class T_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # self.norm = Normalize(mean=[0.1307], std=[0.3081])

        self.img = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)

        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, stride=1, padding=2, bias=False)
        # self.dn = DivisiveNormalization2d(sigma=0.1)
        # self.relu1 = TReLU(32, layer_type="conv2d")
        self.relu1 = torch.nn.ReLU()

        self.conv2 = nn.Conv2d(50, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.relu2 = torch.nn.ReLU()

        self.fc1 = nn.Linear(7 * 7 * 64, 100, bias=False)
        self.relu3 = TReLU(100, layer_type="linear")

        self.fc2 = nn.Linear(100, num_classes, bias=True)

    def set_bias(self, alpha: float, layers: Iterable[Iterable]):
        modules = dict([*self.named_modules()])
        for layer_id, activation_id in layers:
            layer = modules[layer_id]
            activation = modules[activation_id]
            l1_weights = norm_weight(layer.weight, lp_norm=1)
            if layer.weight.ndim == 4:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2) * alpha)
            elif layer.weight.ndim == 2:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0) * alpha)
            activation.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.img(x)
        out = self.conv1(out) / ((self.conv1.weight**2).sum(dim=(1, 2, 3),
                                                            keepdim=True).transpose(0, 1).sqrt()+1e-6)
        out = self.relu1(out)
        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"T_LeNet"


class TT_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.relu_start = TReLU(1, layer_type="conv2d")
        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)
        # self.dn = DivisiveNormalization2d(sigma=0.1)
        self.relu_start.bias = torch.nn.Parameter(0.3*torch.ones_like(self.relu_start.bias))

        self.relu1 = TReLU(32, layer_type="conv2d")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)

        self.relu2 = TReLU(64, layer_type="conv2d")
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)

        self.relu3 = TReLU(1024, layer_type="linear")
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def set_bias(self, alpha: float, layers: Iterable[tuple]):
        modules = dict([*self.named_modules()])
        for layer_id, activation_id in layers:
            layer = modules[layer_id]
            activation = modules[activation_id]
            l1_weights = norm_weight(layer.weight, lp_norm=1)
            if layer.weight.ndim == 4:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2) * alpha)
            elif layer.weight.ndim == 2:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0) * alpha)
            activation.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.relu_start(x)
        out = self.conv1(out)
        out = self.relu1(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"T_LeNet_baseline"


class Leaky_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)

        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)

        self.relu2 = nn.LeakyReLU()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)

        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.relu1(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"Leaky_LeNet"


class BT_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.relu_start = Binarization().apply
        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)
        # self.dn = DivisiveNormalization2d(sigma=0.1)

        self.relu1 = TReLU(32, layer_type="conv2d")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)

        self.relu2 = TReLU(64, layer_type="conv2d")
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)

        self.relu3 = TReLU(1024, layer_type="linear")
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def set_bias(self, alpha: float, layers: Iterable[tuple]):
        modules = dict([*self.named_modules()])
        for layer_id, activation_id in layers:
            layer = modules[layer_id]
            activation = modules[activation_id]
            l1_weights = norm_weight(layer.weight, lp_norm=1)
            if layer.weight.ndim == 4:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2) * alpha)
            elif layer.weight.ndim == 2:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0) * alpha)
            activation.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.relu_start(x)
        out = self.conv1(out)
        out = self.relu1(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"BT_LeNet"


class NT_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # self.relu_start = Binarization().apply
        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.dn = DivisiveNormalization2d(b_type="linf", b_size=(
            5, 5), sigma=0.01, global_suppression=True)

        self.relu1 = TReLU(32, layer_type="conv2d")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)

        self.relu2 = TReLU(64, layer_type="conv2d")
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)

        self.relu3 = TReLU(1024, layer_type="linear")
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def set_bias(self, alpha: float, layers: Iterable[tuple]):
        modules = dict([*self.named_modules()])
        for layer_id, activation_id in layers:
            layer = modules[layer_id]
            activation = modules[activation_id]
            l1_weights = norm_weight(layer.weight, lp_norm=1)
            if layer.weight.ndim == 4:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2) * alpha)
            elif layer.weight.ndim == 2:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0) * alpha)
            activation.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # out = self.relu_start(x)
        out = self.conv1(x)
        out = self.dn(out)
        out = self.relu1(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"NT_LeNet"


class Nl1T_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # self.relu_start = Binarization().apply
        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.dn = DivisiveNormalization2d(b_type="linf", b_size=(
            5, 5), sigma=0.01, global_suppression=True)

        self.relu1 = TReLU(32, layer_type="conv2d")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)

        self.relu2 = TReLU(64, layer_type="conv2d")
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)

        self.relu3 = TReLU(1024, layer_type="linear")
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def set_bias(self, alpha: float, layers: Iterable[tuple]):
        modules = dict([*self.named_modules()])
        for layer_id, activation_id in layers:
            layer = modules[layer_id]
            activation = modules[activation_id]
            l1_weights = norm_weight(layer.weight, lp_norm=1)
            if layer.weight.ndim == 4:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2) * alpha)
            elif layer.weight.ndim == 2:
                activation.bias = torch.nn.Parameter(
                    l1_weights.unsqueeze(0) * alpha)
            activation.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # out = self.relu_start(x)
        out = self.conv1(x)
        out = self.dn(out)
        out = self.relu1(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(self.relu2(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

    def name(self) -> str:
        return f"Nl1T_LeNet"
