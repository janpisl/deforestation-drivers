import os

import pandas as pd
from torch import Tensor
import rasterio

from torch.utils.data import Dataset

class GeoWikiDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['filename'])
        with rasterio.open(img_path) as source:
            image = Tensor(source.read())
        labels = self.img_labels.iloc[idx].drop(['geometry', 'filename', 'sampleid']).to_dict()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels