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
from pytorch_utils.visualization import neuron_visualizer


# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer


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

    model = init_classifier(cfg)

    classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    neuron_visualizer(model=model, layer_name="relu1",
                      filter_num=0, save_loc=cfg.directory + "figs/neurons/")


if __name__ == "__main__":
    main()
