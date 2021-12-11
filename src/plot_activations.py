"""
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from os.path import join
from matplotlib import rc
import matplotlib.pyplot as plt


# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

from mpl_utils.activations import plot_activations

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer, classifier_params_string


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

    model = init_classifier(cfg).to(device)
    model = LayerOutputExtractor_wrapper(model, layer_names=["relu1", "relu2"])

    classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    imgs, _ = test_loader.__iter__().__next__()
    imgs = imgs.to(device)

    _ = model(imgs)

    plot_activations(model.layer_outputs["relu1"][0].detach().cpu(), filepath=cfg.directory +
                     "figs/activations/", file_name=classifier_params_string(model_name=model.name, cfg=cfg))


if __name__ == "__main__":
    main()
