
import numpy as np
import pandas as pd

import pdb

def get_class_counts(dataframe):
    #TODO: extend this so that it works when there is no single majority vote
    array = dataframe.drop(['sampleid', 'filename', 'geometry', 'Unnamed: 0'], axis=1, errors='ignore').to_numpy()
    array = array - array.max(axis=1, keepdims=True)
    array[array == 0] = 1
    array[array < 1] = 0

    assert np.all(array.sum(axis=1) == 1), 'Expected a single majority vote for each class; check what columns are in the dataframe (should be only labels)'

    return array.sum(axis=0) 


def compute_weights(labels):
    """Compute weights to be used in loss function.
    If x votes were cast for one answer,
    assign it x-times higher weight.
    Negative labels have base weight.

    Args:
        labels (Tensor): labels from geowiki dataset

    Returns:
        Tensor: weights
    """
    labels[labels == 0] = 1
    weights = labels/labels.mean()

    return weights


def compute_eval_loss(dataloader, net, criterion, device):
    """Given dataloader, model and loss function, compute loss.

    Args:
        dataloader (torch.utils.data.DataLoader): 
        net (torch.nn.Module): 
        criterion (function): loss function to be used
        device (torch.device): cpu or cuda

    Returns:
        float: loss (scalar)
    """
    test_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs)
        output = output.to(device)
        targets = targets/targets.sum(axis=1, keepdims=True).float()
        loss = criterion(output, targets.float())

        test_loss += loss.item()
    
    return test_loss