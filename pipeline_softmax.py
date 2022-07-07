'''
Example of use:
python pipeline_softmax.py \
    --annotations_path data/tmp/annotations_with_majority_class.csv \
    --image_folder data/tmp/medians/ \
    --batch_size 4
    --wandb
'''

import pdb
import argparse

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
import wandb
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from dataset import GeoWikiDataset
from resnet18 import ResNet18
from get_means_stds import get_means_stds

classes = ['Subsistence agriculture', 'Managed forest/forestry',
       'Pasture', 'Roads/trails/buildings',
       'Other natural disturbances/No tree-loss driver',
       'Commercial agriculture', 'Wildfire (disturbance)',
       'Commercial oil palm or other palm plantations',
       'Mining and crude oil extraction']





def get_image_transform(folder, cropsize=32):

    #skip this to save time in development
    #means, stds = get_means_stds(folder)

    means = [21.08917549,  26.82532432,  50.31721194,  46.70581121, 224.1870091, 162.03204172,  88.59852001]
    stds = [6.85905351,  7.88942854, 10.88926628, 15.86582499, 23.3733194, 32.04417448, 26.91564416]

    return torch.nn.Sequential(
        transforms.CenterCrop(cropsize),
        transforms.Normalize(means, stds),
    )

    

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



def compute_eval_loss(dataloader, net, criterion):
    """Given dataloader, model and loss function, compute loss.

    Args:
        dataloader (torch.utils.data.DataLoader): 
        net (torch.nn.Module): 
        criterion (function): loss function to be used

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


def compute_f1_score(dataloader, net):
    """Compute f1 score
    """
    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs) 
        output = output.to(device)
        pdb.set_trace()
        print('the assert statement below should - but doesnt - sometime fail; there must be some bug')
        assert torch.all(targets.sum(axis=1)) == 1, 'There are multiple classes -> this evaluation doesnt make sense' 
        out = torch.nn.functional.softmax(output, dim=1)
        output_indices = torch.argmax(out, dim=1)
        out[np.arange(len(output_indices)), output_indices] = 1
        out[out != 1] = 0

        all_outputs.append(out)
        all_targets.append(targets)
        
    outputs = torch.concat(all_outputs).cpu()
    targets = torch.concat(all_targets).cpu()
    
    #strict f1 score - only the most voted class counts as correct
    targets = targets - targets.max(axis=1, keepdims=True).values
    targets[targets == 0] = 1
    targets[targets < 1] = 0

    f1_scores = []
    for i in range(9):
        class_f1_score = f1_score(targets[:,i],outputs[:,i])
        f1_scores.append(class_f1_score)

    return f1_scores


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--image_folder', type=str)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)    

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    annotations_path = args.annotations_path
    images_path = args.image_folder

    # Using  weights in loss function results in poor performance, not sure why
    #controls_class_counts = 1023, 254, 308, 53, 76, 104, 16, 139, 19
    #weights_unnorm = torch.Tensor([1/i for i in controls_class_counts]).to(device)
    #weights = weights_unnorm/weights_unnorm.mean()

    batch_size = args.batch_size

    torch.manual_seed(420)

    image_transform = get_image_transform(images_path)

    full_dataset = GeoWikiDataset(
        annotations_file=annotations_path, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True,
        transform=image_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    net = ResNet18(output_sigmoid=False).to(device)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if args.wandb:
        parameters = {
            'batch_size': batch_size,
            'epochs': args.epochs,
            'dataset_size': len(train_dataset)
        }

        wandb.init(project='geowiki_1', 
                entity='janpisl', config=parameters)

        wandb.watch(net, log='all')
    

    for epoch in range(args.epochs):
        epoch_loss = 0
        for inputs, targets in train_dataloader:
            #just to train on a single batch to see what happens
            #if epoch_loss != 0:
            #    continue
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)
            output = output.to(device)
            targets = targets/targets.sum(axis=1, keepdims=True).float()
            loss = criterion(output, targets)
            epoch_loss += loss.item()
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            eval_loss = compute_eval_loss(test_dataloader, net, criterion)
            epoch_loss, eval_loss = epoch_loss*0.2, eval_loss*0.8
            print("Train loss: ", epoch_loss, " Val loss: ", eval_loss)

            train_stats_per_class = compute_f1_score(train_dataloader, net)
            val_stats_per_class = compute_f1_score(test_dataloader, net)

            train_f1_scores = dict(zip(classes, train_stats_per_class))
            train_f1_scores = {'Train -' + k: v for k,v in train_f1_scores.items()}
            

            val_f1_scores = dict(zip(classes, val_stats_per_class))
            val_f1_scores = {'Validation - ' + k: v for k,v in val_f1_scores.items()}         

            avg_f1_score_train = sum(train_stats_per_class)/len(train_stats_per_class)
            avg_f1_score_val = sum(val_stats_per_class)/len(val_stats_per_class)  
            print(f'Average f1 score: train: {avg_f1_score_train}, validation: {avg_f1_score_val}')

            if args.wandb:
                wandb.log({
                        "train_loss": epoch_loss,
                        "validation loss": eval_loss,
                        "Average f1 score (train)":avg_f1_score_train,
                        "Average f1 score (val)":avg_f1_score_val,
                        #"output": wandb.Image(output[0]),
                        #"target": wandb.Image(targets[0]),
                        **train_f1_scores,
                        **val_f1_scores

                })
    
    with torch.no_grad():
        train_stats_per_class = compute_f1_score(train_dataloader, net)
        val_stats_per_class = compute_f1_score(test_dataloader, net)


