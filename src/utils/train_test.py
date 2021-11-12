"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils.loss import saliency_K


def standard_epoch(model, train_loader, optimizer, scheduler=None, verbose: bool = False):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        # loss = 0
        tobe_regularized = {key: value for key, value in model.layer_outputs.items() if key in [
            "relu1"]}

        loss -= 1.0 * torch.mean(saliency_K(features=tobe_regularized,
                                            K=5, saliency_lambda=0.5, dim=1))

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    if verbose:
        print(f"Train loss: {train_loss/train_size:.2f}, Train acc: {train_correct/train_size:.2f}")

    return train_loss/train_size, train_correct/train_size


def standard_test(model, test_loader, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)
    if verbose:
        print(f"Test loss: {test_loss/test_size:.2f}, Test acc: {test_correct/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size


"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""


def standard_epoch(model, train_loader, optimizer, scheduler=None, verbose: bool = False):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        # loss = 0
        tobe_regularized = {key: value for key, value in model.layer_outputs.items() if key in [
            "relu1"]}

        loss -= 1.0 * torch.mean(saliency_K(features=tobe_regularized,
                                            K=5, saliency_lambda=0.1, dim=1))

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    if verbose:
        print(f"Train loss: {train_loss/train_size:.2f}, Train acc: {train_correct/train_size:.2f}")

    return train_loss/train_size, train_correct/train_size


def standard_test(model, test_loader, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)
    if verbose:
        print(f"Test loss: {test_loss/test_size:.2f}, Test acc: {test_correct/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size
