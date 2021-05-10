import torch
import numpy as np

import matplotlib.pyplot as plt


def Loss_Simulator(prediction, target):
    return torch.mean(torch.pow(prediction - target), 2)


def ComputeTestErrorRMSE(prediction, target):
    tmp_MSE = torch.sqrt(torch.mean((prediction - target) ** 2, 1))
    tmp_max = torch.max(target, 1)[0]
    tmp_min = torch.min(target, 1)[0]
    tmp_range = tmp_max - tmp_min
    tmp_error = torch.div(tmp_MSE, tmp_range)
    return tmp_error


def ComputeTestErrorMAE(prediction, target):

    tmp_MAE = torch.mean(torch.abs(prediction - target), 1)
    tmp_error = tmp_MAE

    return tmp_error


def ComputeTestErrorRMAE(prediction, target):
    # Divide error by the range in each sample
    tmp_MAE = torch.mean(torch.abs(prediction - target), 1)
    tmp_max = torch.max(target, 1)[0]
    tmp_min = torch.min(target, 1)[0]
    tmp_range = tmp_max - tmp_min
    tmp_error = torch.div(tmp_MAE, tmp_range)

    return tmp_error


def ComputeTestErrorRAE(prediction, target):
    # Divide absolute error by target nodal value
    tmp_AE = torch.abs(torch.div(prediction - target, target))
    tmp_error = torch.mean(tmp_AE, 1)

    return tmp_error
