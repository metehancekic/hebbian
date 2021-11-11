import torch
import os
import numpy as np


def classifier_params_string(model_name, cfg):
    classifier_params_string = model_name

    classifier_params_string += f"_{cfg.nn.optimizer}"

    classifier_params_string += f"_{cfg.nn.scheduler}"

    if cfg.nn.scheduler == "cyc":
        classifier_params_string += f"_{cfg.nn.lr_max:.4f}"
    else:
        classifier_params_string += f"_{cfg.nn.lr:.4f}"

    classifier_params_string += f"_{cfg.train.regularizer}"

    classifier_params_string += f"_ep_{cfg.train.epochs}"

    return classifier_params_string


def classifier_ckpt_namer(model_name, cfg):

    file_path = cfg.directory + f"checkpoints/classifiers/{cfg.dataset}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".pt"

    return file_path


def classifier_log_namer(model_name, cfg):

    file_path = cfg.directory + f"logs/{cfg.dataset}/"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".log"

    return file_path
