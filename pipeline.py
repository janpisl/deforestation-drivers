import torch

import pandas as pd
import geopandas as gpd
import os
import pdb
import pandas as pd
import numpy as np

import wandb


from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import GeoWikiDataset
from model import Net
from resnet18 import ResNet18

image_transform = torch.nn.Sequential(
    transforms.CenterCrop(32),
    transforms.Normalize((8472.0, 9534.1, 9378.7, 17898.8), (313.3, 431.8,  644.0, 1335.2)),
)


def compute_weights(labels):
    """Compute weights to be used in loss function.
    If x votes were cast for one answer,
    assign it x-times higher weight.
    Negative labels have base weight

    Args:
        labels (Tensor): labels from geowiki dataset

    Returns:
        Tensor: weights
    """
    labels[labels == 0] = 1
    weights = labels/labels.sum()

    return weights


def weighted_loss_function(output, target_dict, loss_func):
    """Compute weighted loss. Each class has a weight equivalent to the
    number of votes, except if there are 0 votes, the weight is 1. All weights
    are then normalized so that they sum to 1.

    Args:
        output (Tensor): output of model
        target_dict (dict): target - for each class, the number of votes
        loss_func (function): loss function to be used

    Returns:
        float: loss (scalar)
    """
    labels = torch.stack([value for key, value in target_dict.items()]).T
    labels_mask = labels.clone()
    labels_mask[labels_mask > 0] = 1

    loss_array = loss_func(output, labels_mask.float())
    weights = compute_weights(labels)
    loss = (weights*loss_array).mean()

    return loss


def compute_eval_loss(dataloader, net, loss_func):

    test_loss = 0
    for inputs, targets in dataloader:
        output = net(inputs)

        labels = torch.stack([value for key, value in targets.items()]).T
        loss = loss_func(output, labels.float())
        test_loss += loss.item()
    
    return test_loss


def compute_stats(dataloader, net, threshold=0.5):
    for inputs, targets in dataloader:
        output = net(inputs) 

        #Set logits over threshold to 0
        output[output >= threshold] = 1
        output[output < threshold] = 0

        labels = torch.stack([value for key, value in targets.items()]).T
        
        outputs_where_true = output[labels == 1]
        outputs_where_false = output[labels == 0]

        TP = outputs_where_true.sum()/len(outputs_where_true)
        FP = outputs_where_false.sum()/len(outputs_where_false)

        TN = (1 - outputs_where_false).sum()/len(outputs_where_false)
        FN = (1 - outputs_where_true).sum()/len(outputs_where_true)

        return TP, FP, TN, FN



if __name__ == '__main__':
    
    #controls = pd.read_csv('data/ILUC_controls_labels.csv')
    #campaign = pd.read_csv('data/ILUC_campaign_labels.csv')


    annotations_path = 'data/controls_labels_processed.csv'
    images_path = 'data/controls_L8_examples/'

    nb_epochs, batch_size =  20, 8
    weighted_loss = False

    torch.manual_seed(420)


    full_dataset = GeoWikiDataset(annotations_file=annotations_path, img_dir=images_path, transform=image_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #net = Net()
    #number_of_params = sum(p.numel() for p in net.parameters())
    net = ResNet18()
    optimizer = torch.optim.Adam(net.parameters())
    
    if weighted_loss:
        loss_func = torch.nn.BCELoss(reduction='none')
    else:
        criterion = torch.nn.BCELoss()


    '''wandb.init(project='geowiki_1', 
            entity='janpisl')

    wandb.watch(net, log='all')'''
    

    for epoch in range(nb_epochs):
        epoch_loss = 0
        for inputs, targets in train_dataloader:
            output = net(inputs)
            if weighted_loss:
                loss = weighted_loss_function(output, targets, loss_func=loss_func)
            else:
                labels = torch.stack([value for key, value in targets.items()]).T
                loss = criterion(output, labels.float())
            epoch_loss += loss.item()
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            eval_loss = compute_eval_loss(test_dataloader, net, criterion)
            TP, FP, TN, FN = compute_stats(train_dataloader, net)
            val_TP, val_FP, val_TN, val_FN = compute_stats(test_dataloader, net)
            print("TP, FP, TN, FN: ", TP, FP, TN, FN)
            print("validation: TP, FP, TN, FN: ", val_TP, val_FP, val_TN, val_FN )

            print("Train loss: ", epoch_loss, " Val loss: ", eval_loss)
            
            '''wandb.log({
                    "train_loss": epoch_loss,
                    "validation loss": eval_loss,
                    #"output": wandb.Image(output[0]),
                    #"target": wandb.Image(targets[0]),
                    "TP": TP, "FP": FP, "TN" : TN, "FN": FN,
                    #"val_TP": val_TP, "val_FP": val_FP, "val_TN" : val_TN, "val_FN": val_FN

            })'''

