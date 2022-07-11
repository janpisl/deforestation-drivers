import os
import pdb

import pandas as pd
from torch import Tensor
import rasterio
from torchvision import transforms
from torch.nn import Sequential
from torch.utils.data import Dataset
from preprocessing import file_exists, has_missing_data, get_file_names, has_single_label


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