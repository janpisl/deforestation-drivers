import torch
from torchmetrics.functional import f1_score, precision_recall

import torch.nn.functional as F
import pdb

CLASSES = ['Subsistence agriculture', 'Managed forest/forestry',
       'Pasture', 'Roads/trails/buildings',
       'Other natural disturbances/No tree-loss driver',
       'Commercial agriculture', 'Wildfire (disturbance)',
       'Commercial oil palm or other palm plantations',
       'Mining and crude oil extraction']


def compute_stats(dataloader, net, device, num_classes=9):
    """Compute f1 score, precision, recall for each class separately
    """
    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs) 
        output = output.to(device)
        out = torch.nn.functional.log_softmax(output, dim=1)
        
        output_classes = torch.argmax(out, dim=1)
        #strict f1 score - only the most voted class counts as correct
        #target_classes = torch.argmax(targets, axis=1)

        all_outputs.append(output_classes)
        all_targets.append(targets)


    outputs = torch.concat(all_outputs)
    targets = torch.concat(all_targets)
            
    f1_scores = f1_score(outputs, targets, average=None, num_classes=num_classes)
    precision_scores, recall_scores = precision_recall(outputs, targets, average=None, num_classes=num_classes)

    return f1_scores, precision_scores, recall_scores


def compute_accuracy(dataloader, net, device):
    """accuracy
    """
    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs) 
        output = output.to(device)
        out = torch.nn.functional.log_softmax(output, dim=1)
        
        output_classes = torch.argmax(out, dim=1)
        #strict f1 score - only the most voted class counts as correct
        #target_classes = torch.argmax(targets, axis=1)

        all_outputs.append(output_classes)
        all_targets.append(targets)


    outputs = torch.concat(all_outputs)
    targets = torch.concat(all_targets)

    total = len(targets)
    
    return ((outputs == targets).unique(return_counts=True)[1][-1]/total).item()
