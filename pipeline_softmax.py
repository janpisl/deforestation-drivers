'''
Example of use:
python pipeline_softmax.py \
    --annotations_path data/tmp/annotations_with_majority_class.csv \
    --image_folder data/seco_campaign_landsat/medians_fixed_naming/ \
    --batch_size 128\
    --weighted_loss\
    --wandb
'''

import pdb
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_class_counts
import wandb
from torchmetrics import F1Score
from torchmetrics.functional import f1_score, precision_recall

from utils import get_class_counts, compute_eval_loss
from dataset import GeoWikiDataset, get_image_transform
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
         missing_vals_dropped, 
         single_label_only, 
         log_wandb, 
         class_weights=None):

    print(f'\n Parameters: weighted loss {weighted_loss}, batch_size {batch_size}, lr {lr}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    net = ResNet18(output_sigmoid=False).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
    
    if weighted_loss:
        print('weighted loss is used')
        assert class_weights is not None, "Class weights must be provided for weighted loss"
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if log_wandb:
        parameters = {
            'batch_size': batch_size,
            'epochs': epochs,
            'dataset_size': len(train_dataset),
            'weighted loss': weighted_loss,
            'learning rate': lr,
            'images dropped if missing values': missing_vals_dropped,
            'images dropped if multiple answers': single_label_only,
            'weight decay': weight_decay
        }

        wandb.init(project='geowiki_1', 
                entity='janpisl', reinit=True, config=parameters)

        wandb.watch(net, log='all')
    

    best_val_f1_score = 0
    best_epoch = 0

    for epoch in range(epochs):

        epoch_loss = train(net, train_dataloader,criterion, optimizer, device)
        
        with torch.no_grad():
            eval_loss = compute_eval_loss(test_dataloader, net, criterion, device)
            
            #Scaling losses based on the train/test split so they are comparable
            epoch_loss, eval_loss = epoch_loss*0.2, eval_loss*0.8
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


            avg_f1_score_train = train_f1_scores.mean()
            avg_f1_score_val = val_f1_scores.mean()
            print(f'Average f1 score: train: {avg_f1_score_train}, validation: {avg_f1_score_val}')

            if avg_f1_score_val > best_val_f1_score:
                best_val_f1_score = avg_f1_score_val
                best_epoch = epoch

            if log_wandb:
                wandb.log({
                        "train_loss": epoch_loss,
                        "validation loss": eval_loss,
                        "Average f1 score (train)":avg_f1_score_train,
                        "Average f1 score (val)":avg_f1_score_val,
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



def get_datasets(annotations_path, images_path, drop_missing_vals, single_label_only, device):

    image_transform = get_image_transform(images_path)

    full_dataset = GeoWikiDataset(
        annotations_file=annotations_path, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
        drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
        single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
        transform=image_transform)

    #I think technically i should only use the train dataset to get class counts
    class_counts = get_class_counts(full_dataset.img_labels)
    weights_unnorm = torch.Tensor([1/i for i in class_counts]).to(device)
    weights = weights_unnorm/weights_unnorm.mean()

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset, weights


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--image_folder', type=str)

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

    annotations_path, images_path, batch_size, epochs, drop_missing_vals, single_label_only, lr, weight_decay, log_wandb = \
        args.annotations_path, args.image_folder, args.batch_size, args.epochs, args.drop_rows_with_missing_vals, args.single_label_only, args.lr, args.weight_decay, args.wandb

    weighted_loss = True if args.weighted_loss is not None else False

    torch.manual_seed(420)

    train_dataset, test_dataset, class_weights = get_datasets(annotations_path, images_path, drop_missing_vals, single_label_only, device)

    main(train_dataset, test_dataset, device, weighted_loss, batch_size, epochs, lr, weight_decay, drop_missing_vals, single_label_only, log_wandb, class_weights)

