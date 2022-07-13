'''
Example of use:
python pipeline_softmax.py \
    --annotations_path data/tmp/small_more_balanced_dst.csv \
    --image_folder data/seco_campaign_landsat/medians_fixed_naming/ \
    --annotations_path_val data/controls_labels_processed.csv \
    --image_folder_val data/seco_controls_l8_2020/medians/ \
    --batch_size 128\
    --drop_rows_with_missing_vals\
    --single_label_only\
    --weight_decay 0.001\
    --weighted_loss\
    --wandb
'''

import pdb
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from torchmetrics import F1Score
from torchmetrics.functional import f1_score, precision_recall

from utils import compute_eval_loss
from dataset import GeoWikiDataset, get_image_transform, get_datasets
from resnet18 import ResNet18

CLASSES = ['Subsistence agriculture', 'Managed forest/forestry',
       'Pasture', 'Roads/trails/buildings',
       'Other natural disturbances/No tree-loss driver',
       'Commercial agriculture', 'Wildfire (disturbance)',
       'Commercial oil palm or other palm plantations',
       'Mining and crude oil extraction']


def compute_stats(dataloader, net, device):
    """Compute f1 score
    """
    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs) 
        output = output.to(device)
        out = torch.nn.functional.softmax(output, dim=1)
        
        output_classes = torch.argmax(out, dim=1)
        #strict f1 score - only the most voted class counts as correct
        target_classes = torch.argmax(targets, axis=1)

        all_outputs.append(output_classes)
        all_targets.append(target_classes)


    outputs = torch.concat(all_outputs)
    targets = torch.concat(all_targets)
    
    f1_scores = f1_score(outputs, targets, average=None, num_classes=len(CLASSES))
    precision_scores, recall_scores = precision_recall(outputs, targets, average=None, num_classes=len(CLASSES))

    return f1_scores, precision_scores, recall_scores



def train(net, dataloader, criterion, optimizer, device):

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



def main(train_dataset, 
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

    net = ResNet18(output_sigmoid=False).to(device)
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

        epoch_loss = train(net, train_dataloader,criterion, optimizer, device)
        
        with torch.no_grad():
            eval_loss = compute_eval_loss(test_dataloader, net, criterion, device)
            
            #Scale losses based on len(test_dataloader) and len(train_dataloader) so they are comparable
            #and multiplying them by 1000 so they have reasonable values
            epoch_loss, eval_loss = 1000*epoch_loss/len(train_dataset), 1000*eval_loss/len(test_dataset)
            print(f"\nEpoch: {epoch}, Train loss: ", epoch_loss, " Val loss: ", eval_loss)

            train_f1_scores, train_precision, train_recall = compute_stats(train_dataloader, net, device)
            val_f1_scores, val_precision, val_recall = compute_stats(test_dataloader, net, device)

            train_f1_dict = dict(zip(CLASSES, train_f1_scores))
            train_f1_dict = {'Train f1 -' + k: v for k,v in train_f1_dict.items()}
            
            train_precision_dict = dict(zip(CLASSES, train_precision))
            train_precision_dict = {'Train precision -' + k: v for k,v in train_precision_dict.items()}           
            
            train_recall_dict = dict(zip(CLASSES, train_recall))
            train_recall_dict = {'Train recall -' + k: v for k,v in train_recall_dict.items()} 

            val_f1_dict = dict(zip(CLASSES, val_f1_scores))
            val_f1_dict = {'Val f1 -' + k: v for k,v in val_f1_dict.items()}
            
            val_precision_dict = dict(zip(CLASSES, val_precision))
            val_precision_dict = {'Val precision -' + k: v for k,v in val_precision_dict.items()}           
            
            val_recall_dict = dict(zip(CLASSES, val_recall))
            val_recall_dict = {'Val recall - ' + k: v for k,v in val_recall_dict.items()} 


            f1_score_train_mean, f1_score_train_median = train_f1_scores.mean(), train_f1_scores.median()
            f1_score_val_mean, f1_score_val_median = val_f1_scores.mean(),  val_f1_scores.median()
            print(f'Mean f1 score: train: {f1_score_train_mean}, validation: {f1_score_val_mean}')

            if f1_score_val_mean > best_val_f1_score:
                best_val_f1_score = f1_score_val_mean
                best_epoch = epoch

            if log_wandb:
                wandb.log({
                        "train_loss": epoch_loss,
                        "validation loss": eval_loss,
                        "Mean f1 score (train)":f1_score_train_mean,
                        "Mean f1 score (val)":f1_score_val_mean,
                        "Median f1 score (train)":f1_score_train_median,
                        "Median f1 score (val)":f1_score_val_median,
                        "Best average f1 score (val)": best_val_f1_score,
                        "Epoch with best val f1 score": best_epoch,
                        #"output": wandb.Image(output[0]),
                        #"target": wandb.Image(targets[0]),
                        **train_f1_dict,
                        **train_precision_dict,
                        **train_recall_dict,
                        **val_f1_dict,
                        **val_precision_dict,
                        **val_recall_dict


                })


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--image_folder', type=str)

    parser.add_argument('--annotations_path_val', type=str, default=None)
    parser.add_argument('--image_folder_val', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--weighted_loss', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--drop_rows_with_missing_vals', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--single_label_only', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)    

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size, epochs, drop_missing_vals, single_label_only, lr, weight_decay, log_wandb = \
        args.batch_size, args.epochs, args.drop_rows_with_missing_vals, args.single_label_only, args.lr, args.weight_decay, args.wandb

    annotations_path, images_path, annotations_path_val, image_folder_val = \
        args.annotations_path, args.image_folder, args.annotations_path_val, args.image_folder_val,

    weighted_loss = True if args.weighted_loss is not None else False

    torch.manual_seed(420)
    np.random.seed(420)


    train_dataset, test_dataset, class_weights = get_datasets(annotations_path, 
                                                              images_path, 
                                                              drop_missing_vals, 
                                                              single_label_only, 
                                                              annotations_path_val, 
                                                              image_folder_val, 
                                                              device)


    log_dict = {
            'batch_size': batch_size,
            'epochs': epochs,
            'dataset_size': len(train_dataset),
            'evaluation dataset size': len(test_dataset),
            'weighted loss': weighted_loss,
            'learning rate': lr,
            'images dropped if missing values': drop_missing_vals,
            'images dropped if multiple answers': single_label_only,
            'weight decay': weight_decay
        }


    main(train_dataset, 
         test_dataset, 
         device, 
         weighted_loss, 
         batch_size, 
         epochs, 
         lr, 
         weight_decay, 
         log_wandb, 
         log_dict, 
         class_weights)



