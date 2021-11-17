"""
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from os.path import join
from matplotlib import rc
import matplotlib.pyplot as plt

import torch.nn.functional as F

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer
from .models.custom_layers import LpConv2d
from .models import LeNet


@hydra.main(config_path="/home/metehan/hebbian/src/configs", config_name="mnist")
def main(cfg: DictConfig) -> None:

    rc('font', ** {
       'family': 'serif',
       'serif': ["Times"]
       })
    rc('text', usetex=True)

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, data_params = init_dataset(cfg)

    lp_norm_extractor = LpConv2d(in_channels=32, out_channels=1,
                                 kernel_size=5, stride=1, padding=2, bias=False, p_norm=2).to(device)
    model_base = LeNet().to(device)
    model_match = init_classifier(cfg).to(device)

    model_base = LayerOutputExtractor_wrapper(model_base, layer_names=["relu1", "conv2"])

    model_match = LayerOutputExtractor_wrapper(model_match, layer_names=["relu1", "conv2"])

    classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    model_match.load_state_dict(torch.load(classifier_filepath))

    base_filepath = cfg.directory + f"checkpoints/classifiers/{cfg.dataset}/" + "LeNet_adam_none_0.0010_none_ep_40.pt"
    model_base.load_state_dict(torch.load(base_filepath))

    nb_cols = 1
    nb_rows = 1
    plt.figure(figsize=(10 * nb_cols, 4 * nb_rows))
    for i in range(nb_cols * nb_rows):
        plt.subplot(nb_rows, nb_cols, i + 1)
        img_index = np.random.choice(50000)
        print(f"image: {img_index},", end=" ")
        img, _ = train_loader.dataset[img_index]
        img = img.to(device)

        _ = model_base(img.unsqueeze(0))
        _ = model_match(img.unsqueeze(0))

        patch_norms_base = lp_norm_extractor(
            F.max_pool2d(model_base.layer_outputs["relu1"], (2, 2)))
        patch_norms_base = torch.repeat_interleave(patch_norms_base, 64, dim=1)

        patch_norms_match = lp_norm_extractor(
            F.max_pool2d(model_match.layer_outputs["relu1"], (2, 2)))
        patch_norms_match = torch.repeat_interleave(patch_norms_match, 64, dim=1)

        base_out = model_base.layer_outputs["conv2"]

        weight_base = (model_base.conv2.weight**2).sum(dim=(1, 2, 3),
                                                       keepdim=True).transpose(0, 1).sqrt()

        base_out /= (patch_norms_base*weight_base + 1e-8)

        match_out = model_match.layer_outputs["conv2"]
        weight_match = (model_match.conv2.weight**2).sum(dim=(1, 2, 3),
                                                         keepdim=True).transpose(0, 1).sqrt()
        match_out /= (patch_norms_match*weight_match + 1e-8)

        # match_patch = match_out.squeeze().detach().cpu().numpy()[:, 10:18, 10:18]
        # base_patch = base_out.squeeze().detach().cpu().numpy()[:, 10:18, 10:18]
        match_patch = match_out[patch_norms_match > 0.1].detach().cpu().numpy()
        base_patch = base_out[patch_norms_base > 0.1].detach().cpu().numpy()

        abs_max = max(np.abs(match_patch).max(), np.abs(match_patch).max())
        xlims = (-abs_max, abs_max)

        bin_edges = np.linspace(*xlims, 50)

        hist, _ = np.histogram(match_patch, bin_edges, density=True)

        color, edgecolor = ("orange", "darkorange")

        plt.bar(
            bin_edges[:-1] + np.diff(bin_edges) / 2,
            hist,
            width=(bin_edges[1] - bin_edges[0]),
            alpha=0.5,
            edgecolor="none",
            color=color,
            )
        plt.step(
            np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
            np.array([0, *hist, 0]),
            label=r"Hebbian model",
            where="pre",
            color=edgecolor,
            )
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_yaxis().set_visible(False)

        hist, _ = np.histogram(base_patch, bin_edges, density=True)

        color, edgecolor = ("steelblue", "steelblue")

        plt.bar(
            bin_edges[:-1] + np.diff(bin_edges) / 2,
            hist,
            width=(bin_edges[1] - bin_edges[0]),
            alpha=0.5,
            edgecolor="none",
            color=color,
            )
        plt.step(
            np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
            np.array([0, *hist, 0]),
            label=r"Base model",
            where="pre",
            color=edgecolor,
            )
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.legend()

    plt.tight_layout()

    os.makedirs(cfg.directory + "figs/", exist_ok=True)
    plt.savefig(join(cfg.directory + 'figs', 'correlations_second_single.pdf'))
    plt.close()


if __name__ == "__main__":
    main()
