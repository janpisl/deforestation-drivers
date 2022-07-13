import os
import pdb

import pandas as pd
from torch import Tensor
import rasterio
from torch import utils
from torchvision import transforms
from torch.nn import Sequential
from torch.utils.data import Dataset
from preprocessing import file_exists, has_missing_data, get_file_names, has_single_label
from utils import get_class_counts

class GeoWikiDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 drop_rows_with_missing_file=True, 
                 drop_rows_with_nan_data=False, 
                 single_label_rows_only=False, 
                 transform=None, 
                 target_transform=None):
    
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels['filename'] = get_file_names(self.img_labels, 'sampleid')

        if drop_rows_with_missing_file:
            size = len(self.img_labels)
            self.img_labels = self.img_labels.loc[file_exists(self.img_labels, img_dir)].reset_index(drop=True)
            print(f'Dropped {size - len(self.img_labels)} rows where the corresponding file was not found. Rows remaining: {len(self.img_labels)}')
        if single_label_rows_only:
            size = len(self.img_labels)
            self.img_labels = self.img_labels.loc[has_single_label(self.img_labels)].reset_index(drop=True)
            print(f'Dropped {size - len(self.img_labels)} rows with two or more labels. Rows remaining: {len(self.img_labels)}')
        if drop_rows_with_nan_data:
            size = len(self.img_labels)
            self.img_labels = self.img_labels.loc[~has_missing_data(self.img_labels, img_dir)].reset_index(drop=True)
            print(f'Dropped {size - len(self.img_labels)} rows where the corresponding file contained missing data. Rows remaining: {len(self.img_labels)}')  
                 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['filename'])
        with rasterio.open(img_path) as source:
            image = Tensor(source.read())
        labels = self.img_labels.iloc[idx].drop(['geometry', 'filename', 'sampleid'], errors='ignore').to_dict()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        labels = Tensor([value for _, value in labels.items()])

        return image, labels



def get_image_transform(folder, cropsize=32):

    #skip this to save time in development
    #means, stds = get_means_stds(folder)

    means = [21.08917549,  26.82532432,  50.31721194,  46.70581121, 224.1870091, 162.03204172,  88.59852001]
    stds = [6.85905351,  7.88942854, 10.88926628, 15.86582499, 23.3733194, 32.04417448, 26.91564416]

    return Sequential(
        transforms.CenterCrop(cropsize),
        transforms.Normalize(means, stds),
    )



def get_datasets(annotations_path, 
                 images_path, 
                 drop_missing_vals, 
                 single_label_only,
                 annotations_path_test, 
                 image_folder_test,
                 device):

    image_transform = get_image_transform(images_path)

    full_dataset = GeoWikiDataset(
        annotations_file=annotations_path, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
        drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
        single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
        transform=image_transform)


    #I think technically i should only use the train dataset to get class counts
    #TODO: weights not implemented because get_class_counts() currently 
    #requires a single majority class
    #class_counts = get_class_counts(full_dataset.img_labels)
    #weights_unnorm = Tensor([1/i for i in class_counts]).to(device)
    #weights = weights_unnorm/weights_unnorm.mean()
    weights = None

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = utils.data.random_split(full_dataset, [train_size, test_size])


    # If separate validation set is provided, it is used for validation
    # Note that the test_dataset from random_split is not used at all and can be used for final evaluation
    if annotations_path_test and image_folder_test:
        test_dataset = GeoWikiDataset(
            annotations_file=annotations_path_test, 
            img_dir=image_folder_test, 
            drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
            drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
            single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
            transform=image_transform)   
        

    return train_dataset, test_dataset, weights