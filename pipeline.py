'''
Example of use:
python pipeline_v2.py config/config.json

'''




import sys
import json
import pdb
import argparse
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from utils import compute_eval_loss, parse_boolean, set_seed
from dataset import get_datasets
from model import ResNet18, get_resnet18_pytorch
from eval import compute_stats, CLASSES




def train_one_epoch(net, dataloader, criterion, optimizer, device):

    epoch_loss = 0
    count = 0
    for inputs, targets in dataloader:
        #just to train on a few batches to see what happens
        #count+=1
        #if count > 10:
        #    break
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs)
        output = output.to(device)
        targets = targets/targets.sum(axis=1, keepdims=True).float()
        loss = criterion(output, targets)
        epoch_loss += loss.item()
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
    
    return epoch_loss



def train(train_dataset, 
         test_dataset,
         device, 
         weighted_loss, 
         batch_size, 
         epochs, 
         lr, 
         weight_decay,
         log_wandb, 
         config_log,
         class_weights=None):

    print(f'\n Parameters: weighted loss {weighted_loss}, batch_size {batch_size}, lr {lr}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #net = ResNet18(output_sigmoid=False).to(device)
    net = get_resnet18_pytorch().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
    
    if weighted_loss:
        assert class_weights is not None, "Class weights must be provided for weighted loss"
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:   
        criterion = torch.nn.CrossEntropyLoss()

    if log_wandb:

        wandb.init(project='geowiki_1', 
                entity='janpisl', reinit=True, config=config_log)

        wandb.watch(net, log='all')


    best_val_f1_score = 0
    best_epoch = 0

    for epoch in range(epochs):

        epoch_loss = train_one_epoch(net, train_dataloader,criterion, optimizer, device)
        
        with torch.no_grad():

            eval_loss = compute_eval_loss(test_dataloader, net, criterion, device)

            #Scale losses based on len(test_dataloader) and len(train_dataloader) so they are comparable
            #and multiplying them by 1000 so they have reasonable values
            epoch_loss, eval_loss = 1000*epoch_loss/len(train_dataset), 1000*eval_loss/len(test_dataset)
            print(f"\nEpoch: {epoch}, Train loss: ", epoch_loss, " Val loss: ", eval_loss)

            #train_f1_scores, train_precision, train_recall = compute_stats(train_dataloader, net, device)
            val_f1_scores, val_precision, val_recall = compute_stats(test_dataloader, net, device)

            #train_f1_dict = dict(zip(CLASSES, train_f1_scores))
            #train_f1_dict = {'Train f1 -' + k: v for k,v in train_f1_dict.items()}
            
            #train_precision_dict = dict(zip(CLASSES, train_precision))
            #train_precision_dict = {'Train precision -' + k: v for k,v in train_precision_dict.items()}           
            
            #train_recall_dict = dict(zip(CLASSES, train_recall))
            #train_recall_dict = {'Train recall -' + k: v for k,v in train_recall_dict.items()} 

            val_f1_dict = dict(zip(CLASSES, val_f1_scores))
            val_f1_dict = {'Val f1 -' + k: v for k,v in val_f1_dict.items()}
            
            val_precision_dict = dict(zip(CLASSES, val_precision))
            val_precision_dict = {'Val precision -' + k: v for k,v in val_precision_dict.items()}           
            
            val_recall_dict = dict(zip(CLASSES, val_recall))
            val_recall_dict = {'Val recall - ' + k: v for k,v in val_recall_dict.items()} 


            #f1_score_train_mean, f1_score_train_median = train_f1_scores.mean(), train_f1_scores.median()
            f1_score_val_mean, f1_score_val_median = val_f1_scores.mean(),  val_f1_scores.median()
            #print(f'Mean f1 score: train: {f1_score_train_mean}, validation: {f1_score_val_mean}')

            if f1_score_val_mean > best_val_f1_score:
                best_val_f1_score = f1_score_val_mean
                best_epoch = epoch

            if log_wandb:
                wandb.log({
                        "train_loss": epoch_loss,
                        "validation loss": eval_loss,
                        #"Mean f1 score (train)":f1_score_train_mean,
                        "Mean f1 score (val)":f1_score_val_mean,
                        #"Median f1 score (train)":f1_score_train_median,
                        "Median f1 score (val)":f1_score_val_median,
                        "Best average f1 score (val)": best_val_f1_score,
                        "Epoch with best val f1 score": best_epoch,
                        #**train_f1_dict,
                        #**train_precision_dict,
                        #**train_recall_dict,
                        **val_f1_dict,
                        **val_precision_dict,
                        **val_recall_dict


                })



def main(config):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log_wandb = parse_boolean(config['wandb']['log_to_wandb'])
    weighted_loss = parse_boolean(config['search']['weighted_loss'])
    single_label_only = parse_boolean(config['search']['single_label_only'])
    drop_missing_vals = parse_boolean(config['search']['drop_rows_with_missing_vals'])
    majority_label_only = parse_boolean(config['search']['majority_label_only'])
    
    set_seed(config['seed'])

    train_dataset, val_dataset, test_dataset, controls_dataset, class_weights = \
                                                 get_datasets(config['data']['annotations'], 
                                                              config['data']['image_folder'], 
                                                              drop_missing_vals, 
                                                              single_label_only, 
                                                              majority_label_only,
                                                              config['data']['controls']['annotations'], 
                                                              config['data']['controls']['image_folder'])

    config['search']['class_weights'] = class_weights
    config['search']['train dataset size'] = len(train_dataset)
    config['search']['evaluation dataset size'] = len(val_dataset)
    config['search']['model'] = 'torchvision.models.ResNet18(pretrained=False)'

    train(train_dataset, 
         val_dataset, 
         device, 
         weighted_loss, 
         config['search']['batch_size'], 
         config['search']['epochs'], 
         config['search']['learning_rate'], 
         config['search']['weight_decay'], 
         log_wandb, 
         config['search'], 
         class_weights)    


if __name__ == '__main__':

    
    with open(sys.argv[1]) as src:
        config = json.load(src)

    keys, values = zip(*config['search'].items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for variant in permutations_dicts:
        config['search'] = variant
        main(config)





