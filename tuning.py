
import pdb
import argparse

import torch

from pipeline_softmax import get_datasets, main



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--image_folder', type=str)

    parser.add_argument('--epochs', type=int, default=10)   
    parser.add_argument('--drop_rows_with_missing_vals', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--single_label_only', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)    

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(420)


    annotations_path, images_path, epochs, drop_missing_vals, single_label_only, log_wandb = \
        args.annotations_path, args.image_folder, args.epochs, args.drop_rows_with_missing_vals, args.single_label_only, args.wandb

    train_dataset, test_dataset, class_weights = get_datasets(annotations_path, images_path, drop_missing_vals, single_label_only, device)

    for weighted_loss in [False]:
        for batch_size in [128]:
            for lr in [0.0005, 0.001, 0.002, 0.005, 0.01]:
                for weight_decay in [0, 0.01, 0.1]:
                    try:
                        main(train_dataset, test_dataset, device, weighted_loss, batch_size, epochs, lr, weight_decay, drop_missing_vals, single_label_only, log_wandb, class_weights)
                    except Exception as e:
                        print(e)
