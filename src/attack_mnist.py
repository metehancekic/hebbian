"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

# MPL utils
from mpl_utils import save_gif

# Initializers
from .init import *

from .utils.train_test import standard_epoch, standard_test
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils.parsers import parse_regularizer


@hydra.main(config_path="/home/metehan/hebbian/src/configs", config_name="mnist")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, data_params = init_dataset(cfg)
    model = init_classifier(cfg).to(device)

    logger = init_logger(cfg, model.name)

    if not cfg.no_tensorboard:
        writer = init_tensorboard(cfg, model.name)
    else:
        writer = None

    logger.info(model)

    optimizer, scheduler = init_optimizer_scheduler(
        cfg, model, len(train_loader), printer=logger.info, verbose=True)

    _ = count_parameter(model=model, logger=logger.info, verbose=True)

    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))
    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM,
                   )

    attack_params = {
        "norm": cfg.attack.norm,
        "eps": cfg.attack.epsilon,
        "alpha": cfg.attack.alpha,
        "step_size": cfg.attack.step_size,
        "num_steps": cfg.attack.num_steps,
        "random_start": (
            cfg.attack.random and cfg.attack.num_restarts > 1
        ),
        "num_restarts": cfg.attack.num_restarts,
        "EOT_size": cfg.attack.EOT_size,
    }

    if "CWlinf" in cfg.attack.loss:
        loss_function = "carlini_wagner"
    else:
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=PGD,
        attack_args=dict(
            net=model, data_params=data_params, attack_params=attack_params, loss_function=loss_function
        ),
    )
    
    whitebox_test(model=model, test_loader=test_loader, adversarial_args=adversarial_args, verbose=True, progress_bar=True)


if __name__ == "__main__":
    main()
