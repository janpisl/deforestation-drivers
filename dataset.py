import os
import pdb
from re import L

import utm
import numpy as np
import pandas as pd
import geopandas as gpd
from torch import Tensor
import rasterio
from torch import utils
from torchvision import transforms
from torch.nn import Sequential
from torch.utils.data import Dataset
from preprocessing import file_exists, has_majority_label, has_missing_data, get_file_names, has_single_label
from utils import get_class_counts

CLASSES = ['Subsistence agriculture', 'Managed forest/forestry',
       'Pasture', 'Roads/trails/buildings',
       'Other natural disturbances/No tree-loss driver',
       'Commercial agriculture', 'Wildfire (disturbance)',
       'Commercial oil palm or other palm plantations',
       'Mining and crude oil extraction']


class GeoWikiDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 drop_rows_with_missing_file=True, 
                 drop_rows_with_nan_data=False, 
                 majority_label_rows_only=False,
                 single_label_rows_only=False, 
                 undersample=False,
                 transform=None, 
                 target_transform=None):
    
        if isinstance(annotations_file, pd.DataFrame):
            self.img_labels = annotations_file
        else:
            self.img_labels = pd.read_csv(annotations_file)

        if undersample:
            print("An implementation of undersample is being used that is fast but may not result in the exact number of classes.")
            self.img_labels = get_balanced_classes(self.img_labels).reset_index(drop=True)

        self.img_labels['filename'] = get_file_names(self.img_labels, 'sampleid')
        if drop_rows_with_missing_file:
            size = len(self.img_labels)
            self.img_labels = self.img_labels.loc[file_exists(self.img_labels, img_dir)].reset_index(drop=True)
            print(f'Dropped {size - len(self.img_labels)} rows where the corresponding file was not found. Rows remaining: {len(self.img_labels)}')
       
        if single_label_rows_only:
            size = len(self.img_labels)
            self.img_labels = self.img_labels.loc[has_single_label(self.img_labels)].reset_index(drop=True)
            print(f'Dropped {size - len(self.img_labels)} rows with two or more labels. Rows remaining: {len(self.img_labels)}')
       
        if majority_label_rows_only:
            if not single_label_rows_only: #If this was applied, there are no rows without majority label
                size = len(self.img_labels)
                self.img_labels = self.img_labels.loc[has_majority_label(self.img_labels)].reset_index(drop=True)
                print(f'Dropped {size - len(self.img_labels)} rows with two or more labels with equal number of votes. Rows remaining: {len(self.img_labels)}')
       
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

        labels = self.img_labels.iloc[idx].to_dict()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        labels = Tensor([value for key, value in labels.items() if key in CLASSES])

        return image, labels



def get_balanced_classes(df):
    
    labels = df.drop([col for col in df.columns if col not in CLASSES], axis=1)

    classes, counts = np.unique(labels.idxmax(axis=1), return_counts=True)
    min_count = counts.min()
    indices = []
    for _class in CLASSES:
        try:
            class_samples = labels.loc[labels.idxmax(axis=1) == _class].sample(min_count).index
        except ValueError:
            class_samples = labels.loc[labels.idxmax(axis=1) == _class].index
        indices.append(class_samples.tolist())

    def flatten(xss):
        return [x for xs in xss for x in xs]

    indices = flatten(indices)

    return df.loc[indices]


def get_image_transform(folder, cropsize=32):

    #skip this to save time in development
    #means, stds = get_means_stds(folder)

    means = [21.08917549,  26.82532432,  50.31721194,  46.70581121, 224.1870091, 162.03204172,  88.59852001]
    stds = [6.85905351,  7.88942854, 10.88926628, 15.86582499, 23.3733194, 32.04417448, 26.91564416]

    return Sequential(
        transforms.CenterCrop(cropsize),
        transforms.Normalize(means, stds),
    )


def geospatial_data_split(df, method='degree'):
    #TODO:
    # 2 parameters
    #  - size of one stripe if 'degree'
    #  - train/val/test split ratio
    if not method in ['degree', 'utm']:
        raise NotImplementedError("Supported split method are 'degree' and 'utm' ")
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    
    if method == 'degree':
        df['stripe'] = df.geometry.apply(lambda geom: round(geom.x)) 
    elif method == 'utm':
        df['stripe'] = df.geometry.apply(lambda geom: np.sign(geom.y)*(utm.from_latlon(geom.y, geom.x)[2]))
    df['split'] = None
    # Split is 60/20/20
    split = ['Tr', 'Tr', 'Tr', 'Val', 'T']

    for i, stripe in enumerate(np.unique(df['stripe'], return_counts=True)[0]):
        stripes = df.loc[df['stripe'] == stripe]
        df.loc[stripes.index, 'split'] = split[i%len(split)]

    
    return df.loc[df.split == 'Tr'].copy(), df.loc[df.split == 'Val'].copy(), df.loc[df.split == 'T'].copy()



def get_datasets(annotations_path, 
                 images_path, 
                 drop_missing_vals, 
                 majority_label_only,
                 single_label_only,
                 undersample,
                 controls_annotations_path=None,
                 controls_image_path=None):

    image_transform = get_image_transform(images_path)

    train_annotations, val_annotations, test_annotations = geospatial_data_split(pd.read_csv(annotations_path))

    print("\nProcessing training dataset")
    train_dataset = GeoWikiDataset(
        annotations_file=train_annotations, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
        drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
        majority_label_rows_only=majority_label_only, #Only use rows where one class has more votes than any other
        single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
        undersample=undersample, #If True, class balance is forced by only using as many examples from each class that the rarest class has
        transform=image_transform)

    print("\nProcessing validation dataset")
    val_dataset = GeoWikiDataset(
        annotations_file=val_annotations, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
        drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
        majority_label_rows_only=True, #This is false for evaluation in order to compute statistics like Prec/Recall/F1-score 
        single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
        transform=image_transform)

    print("\nProcessing test dataset")
    test_dataset = GeoWikiDataset(
        annotations_file=test_annotations, 
        img_dir=images_path, 
        drop_rows_with_missing_file=True, #This will be always True but keeping it here for explicity
        drop_rows_with_nan_data=drop_missing_vals, #Drop row if any pixel in corresp. image has 0s across all bands
        majority_label_rows_only=True, #This is false for evaluation in order to compute statistics like Prec/Recall/F1-score 
        single_label_rows_only=single_label_only, #Only use rows where all votes are for one class
        transform=image_transform)

    #I think technically i should only use the train dataset to get class counts
    #TODO: weights not implemented because get_class_counts() currently 
    #requires a single majority class
    #class_counts = get_class_counts(full_dataset.img_labels)
    #weights_unnorm = Tensor([1/i for i in class_counts]).to(device)
    #weights = weights_unnorm/weights_unnorm.mean()
    weights = None

    if controls_annotations_path and controls_image_path:
        print("\nProcessing controls dataset")
        controls_dataset = GeoWikiDataset(
            annotations_file=controls_annotations_path, 
            img_dir=controls_image_path, 
            # The following params are hardcoded so the evaluation stays the same throughout
            # different experiments. Single_label_rows should only appear in the validation set but as a sanity check
            #I keep it to True
            drop_rows_with_missing_file=True, 
            drop_rows_with_nan_data=True, 
            majority_label_rows_only=True, #This is false for evaluation in order to compute statistics like Prec/Recall/F1-score 
            single_label_rows_only=True,
            transform=image_transform)
    else:
        controls_dataset = None


    return train_dataset, val_dataset, test_dataset, controls_dataset, weights