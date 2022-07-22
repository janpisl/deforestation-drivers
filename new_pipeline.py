from __future__ import print_function
import json 
import argparse
import itertools
import argparse
import sys
import matplotlib.pyplot as plt
import pdb

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import f1_score, precision_recall
from skimage.exposure import rescale_intensity
import numpy as np

from model import ResNet18, get_resnet18_pytorch, get_vgg, get_squeezenet
from dataset import get_datasets
from eval import compute_stats, compute_accuracy
from utils import parse_boolean, set_seed


CLASSES = ['Subsistence agriculture', 'Managed forest/forestry',
       'Pasture', 'Roads/trails/buildings',
       'Other natural disturbances/No tree-loss driver',
       'Commercial agriculture', 'Wildfire (disturbance)',
       'Commercial oil palm or other palm plantations',
       'Mining and crude oil extraction']


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            


def evaluate(model, device, dataloader, dataset_name, num_classes=9):
    print(f"Evaluation on {dataset_name} set")
    model.eval()
    loss = 0
    correct = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_predictions.append(pred)
            all_targets.append(target.view_as(pred))


        loss /= len(dataloader.dataset)
        outputs = torch.concat(all_predictions)
        targets = torch.concat(all_targets)
        
        f1_scores = f1_score(outputs, targets, average=None, num_classes=num_classes)
        precision_scores, recall_scores = precision_recall(outputs, targets, average=None, num_classes=num_classes)

        accuracy = 100. * correct / len(dataloader.dataset)
        f1_scores = [int(round(100*i)) for i in f1_scores.cpu().tolist()]
        precisions = [int(round(100*i)) for i in precision_scores.cpu().tolist()]
        recalls = [int(round(100*i)) for i in recall_scores.cpu().tolist()]

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(dataloader.dataset),
            accuracy))

        f1_dict = dict(zip(CLASSES, f1_scores))
        f1_dict = {f'{dataset_name} f1 -' + k: v for k,v in f1_dict.items()}

        precision_dict = dict(zip(CLASSES, precisions))
        precision_dict = {f'{dataset_name} precision -' + k: v for k,v in precision_dict.items()}           

        recall_dict = dict(zip(CLASSES, recalls))
        recall_dict = {f'{dataset_name} recall -' + k: v for k,v in recall_dict.items()} 


    return loss, accuracy, f1_dict, precision_dict, recall_dict

    


def main(config, device):
    
    log_wandb = parse_boolean(config['wandb']['log_to_wandb'])
    single_label_only = parse_boolean(config['search']['single_label_only'])
    drop_missing_vals = parse_boolean(config['search']['drop_rows_with_missing_vals'])
    majority_label_only = parse_boolean(config['search']['majority_label_only'])
    undersample = parse_boolean(config['search']['undersample'])
    augmentation = parse_boolean(config['search']['augmentation'])    
    randomize_train_labels = parse_boolean(config['search']['randomize_train_labels'])    


    train_kwargs = {'batch_size': config['search']['batch_size']}
    test_kwargs = {'batch_size': 1000}
    if torch.cuda.is_available():
        cuda_kwargs = {#'num_workers': 1,
                       #'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    '''transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)'''



    train_dataset, val_dataset, test_dataset, controls_dataset, class_weights = \
                                                 get_datasets(config['data']['annotations'], 
                                                              config['data']['image_folder'], 
                                                              drop_missing_vals=drop_missing_vals, 
                                                              majority_label_only=majority_label_only,
                                                              single_label_only=single_label_only,
                                                              undersample=undersample,
                                                              augmentation=augmentation,
                                                              randomize_train_labels=randomize_train_labels,
                                                              controls_annotations_path=config['data']['controls']['annotations'], 
                                                              controls_image_path=config['data']['controls']['image_folder'])
    

    config['search']['train dataset size'] = len(train_dataset)
    config['search']['evaluation dataset size'] = len(val_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    if controls_dataset is not None:
        controls_loader = torch.utils.data.DataLoader(controls_dataset, **test_kwargs)

    #model = Net().to(device)
    #model = ResNet18(in_channels=7, classes=9).to(device)
    model = get_resnet18_pytorch(in_channels=7, output_classes=9).to(device)
    #model = get_vgg(in_channels=7, output_classes=9).to(device)
    #model = get_squeezenet(in_channels=7, output_classes=9).to(device)
    config['search']['model'] = model.__class__.__name__

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    config['search']['Model parameters'] = count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=config['search']['learning_rate'])

    if log_wandb:
        wandb.init(project=config['wandb']['project'], 
                entity=config['wandb']['entity'], reinit=True, config=config['search'])

        wandb.watch(model, log='all', log_freq=4)

    best_val_accuracy = 0
    best_val_accuracy_epoch = 0
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, config['search']['epochs'] + 1):
        train(model, device, train_loader, optimizer, epoch)
        with torch.no_grad():

            train_loss, train_accuracy, train_f1_dict, train_precision_dict, train_recall_dict = evaluate(model, device, train_loader, 'Train', num_classes=9)
            val_loss, val_accuracy, val_f1_dict, val_precision_dict, val_recall_dict = evaluate(model, device, test_loader, 'Validation', num_classes=9)
            if controls_dataset is not None:
                controls_loss, controls_accuracy, controls_f1_dict, controls_precision_dict, controls_recall_dict = evaluate(model, device, controls_loader, "Controls", num_classes=9)


            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_accuracy_epoch = epoch

            if log_wandb:
                wandb.log({
                    "Train loss": train_loss,
                    "Train accuracy": train_accuracy,
                    #**train_f1_dict,
                    #**train_precision_dict,
                    #**train_recall_dict,

                    "Validation loss": val_loss,
                    "Validation accuracy": val_accuracy,
                    "Best val accuracy": best_val_accuracy,
                    "Best val accuracy epoch": best_val_accuracy_epoch,
                    #**val_f1_dict,
                    #**val_precision_dict,
                    #**val_recall_dict,

                    "Controls loss": controls_loss,
                    "Controls accuracy": controls_accuracy,
                    #**controls_f1_dict,
                    #**controls_precision_dict,
                    #**controls_recall_dict,

                })

        #scheduler.step()


if __name__ == '__main__':

    
    with open(sys.argv[1]) as src:
        config = json.load(src)

    keys, values = zip(*config['search'].items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(config['seed'])

    for variant in permutations_dicts:
        config['search'] = variant
        main(config, device)
