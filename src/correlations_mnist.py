"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import matplotlib.pyplot as plt

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

# MPL utils
from mpl_utils import save_gif

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer


@hydra.main(config_path="/home/metehan/hebbian/src/configs", config_name="mnist")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, data_params = init_dataset(cfg)

    model_base = init_classifier(cfg).to(device)
    model_match = init_classifier(cfg).to(device)

    classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    model_match.load_state_dict(torch.load(classifier_filepath))

    base_filepath = cfg.directory + f"checkpoints/classifiers/{cfg.dataset}/" + "T_LeNet_adam_none_0.0010_none_ep_40.pt"
    model_base.load_state_dict(torch.load(base_filepath))

    nb_cols = 2
    nb_rows = 5
    plt.figure(figsize=(10 * nb_cols, 4 * nb_rows))
    for i in range(nb_cols * nb_rows):
        plt.subplot(nb_rows, nb_cols, i + 1)
        img_index = np.random.choice(50000)
        print(f"image: {img_index},", end=" ")
        img, _ = train_loader.dataset[img_index]
        img = img.to(device)

        base_out = model_base.conv1(img.unsqueeze(0))
        base_out /= ((model_base.conv1.weight**2).sum(dim=(1, 2, 3),
                                                      keepdim=True).transpose(0, 1).sqrt()+1e-6)

        match_out = model_match.conv1(img.unsqueeze(0))
        match_out /= ((model_match.conv1.weight**2).sum(dim=(1, 2, 3),
                                                        keepdim=True).transpose(0, 1).sqrt()+1e-6)

        breakpoint()

        patch_index = (np.random.choice(range(1, 28, 2)), np.random.choice(range(1, 28, 2)))

        print(f"patch: {patch_index}")
        match_patch = match_out.squeeze().detach().cpu().numpy()[:, :, patch_index]
        base_patch = base_out.squeeze().detach().cpu().numpy()[:, :, patch_index]

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

    plt.tight_layout()

    os.makedirs(cfg.directory + "figs/", exist_ok=True)
    plt.savefig(join('figs', 'correlations_normalized.pdf'))
    plt.close()


if __name__ == "__main__":
    main()