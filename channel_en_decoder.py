import torch
import torch.nn as nn


def dense(input_size, output_size):  # using dense layer : dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )