import pdb


import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
import wandb
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from dataset import GeoWikiDataset
from resnet18 import ResNet18

image_transform = torch.nn.Sequential(
    transforms.CenterCrop(32),
    transforms.Normalize((8472.0, 9534.1, 9378.7, 17898.8), (313.3, 431.8,  644.0, 1335.2)),
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


def weighted_loss_function(loss_func, output, targets):
    """Compute weighted loss. Each class has a weight equivalent to the
    number of votes, except if there are 0 votes, the weight is 1. All weights
    are then normalized so that they sum to 1.

    Args:
        output (Tensor): output of model
        targets (dict): 
        loss_func (function): loss function to be used

    Returns:
        float: loss (scalar)
    """
    labels_mask = targets.clone()
    labels_mask[labels_mask > 0] = 1

    loss_array = loss_func(output, labels_mask.float())
    weights = compute_weights(targets)
    loss = (weights*loss_array).mean()

    return loss


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

        loss = criterion(output, targets.float())

        test_loss += loss.item()
    
    return test_loss


def compute_f1_score(dataloader, net, threshold=0.5):
    """Compute f1 score
    """
    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs) 
        output = output.to(device)

        #Set logits over threshold to 0
        output[output >= threshold] = 1
        output[output < threshold] = 0

        all_outputs.append(output)
        all_targets.append(targets)
        
    outputs = torch.concat(all_outputs).cpu()
    targets = torch.concat(all_targets).cpu()

    f1_scores = []
    for i in range(9):
        class_f1_score = f1_score(targets[:,i],outputs[:,i])
        f1_scores.append(class_f1_score)

    return f1_scores


if __name__ == '__main__':
    
    #controls = pd.read_csv('data/ILUC_controls_labels.csv')
    #campaign = pd.read_csv('data/ILUC_campaign_labels.csv')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    annotations_path = 'data/controls_labels_processed.csv'
    images_path = 'data/controls_L8_examples/'

    nb_epochs, batch_size =  5, 32
    weighted_loss = True


    torch.manual_seed(420)

    full_dataset = GeoWikiDataset(annotations_file=annotations_path, img_dir=images_path, transform=image_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #net = Net()
    #number_of_params = sum(p.numel() for p in net.parameters())
    net = ResNet18().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    
    if weighted_loss:
        loss_func = torch.nn.BCELoss(reduction='none')
        criterion = partial(weighted_loss_function, loss_func)
    else:
        criterion = torch.nn.BCELoss()


    '''wandb.init(project='geowiki_1', 
            entity='janpisl')

    wandb.watch(net, log='all')'''
    

    for epoch in range(nb_epochs):
        epoch_loss = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)
            output = output.to(device)
            loss = criterion(output, targets.float())
            epoch_loss += loss.item()
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            eval_loss = compute_eval_loss(test_dataloader, net, criterion)
            epoch_loss, eval_loss = epoch_loss*0.2, eval_loss*0.8
            print("Train loss: ", epoch_loss, " Val loss: ", eval_loss)
            
            '''wandb.log({
                    "train_loss": epoch_loss,
                    "validation loss": eval_loss,
                    #"output": wandb.Image(output[0]),
                    #"target": wandb.Image(targets[0]),

            })'''
    
    with torch.no_grad():
        train_stats_per_class = compute_f1_score(train_dataloader, net)
        val_stats_per_class = compute_f1_score(test_dataloader, net)

    pdb.set_trace()
    print()

