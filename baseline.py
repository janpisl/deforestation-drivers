"""
Example of use
python baseline.py \
    --annotations_path data/tmp/annotations_with_majority_class.csv \
    --image_folder data/seco_campaign_landsat/medians_fixed_naming/ \
    --drop_rows_with_missing_vals\
    --single_label_only\
"""


import argparse
import pdb 

import pandas as pd
import numpy as np
import torch
from dataset import GeoWikiDataset
from pipeline_softmax import get_datasets
from feature_extractor import feature_extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def normalize(df):
    X = df[[col for col in df.columns if col != 'target']]
    y = df['target']
    means = X.describe().loc[['mean']].iloc[0]
    stds = X.describe().loc[['std']].iloc[0]
    X = (X - means)/stds
    X['target'] = y

    return X


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)

    parser.add_argument('--train_features_path', type=str, default=None)
    parser.add_argument('--test_features_path', type=str, default=None)


    parser.add_argument('--drop_rows_with_missing_vals', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--single_label_only', action=argparse.BooleanOptionalAction)    

    args = parser.parse_args()

    device = torch.device('cpu')

    annotations_path, images_path, drop_missing_vals, single_label_only = \
        args.annotations_path, args.image_folder, args.drop_rows_with_missing_vals, args.single_label_only, 

    test_features_path, train_features_path = args.test_features_path, args.train_features_path

    torch.manual_seed(420)

    if test_features_path and train_features_path:
        print("Reading features from file")
        train_feats = pd.read_csv(train_features_path)
        test_feats = pd.read_csv(test_features_path)
    else:
        print("Extracting features")
        train_dataset, test_dataset, _ = get_datasets(annotations_path, images_path, drop_missing_vals, single_label_only)

        train_feats = feature_extractor([train_dataset[i] for i in range(len(train_dataset))])
        test_feats = feature_extractor([test_dataset[i] for i in range(len(test_dataset))])

        train_feats = normalize(train_feats)
        test_feats = normalize(test_feats)


    X = train_feats[[col for col in train_feats.columns if col != 'target']]
    y = train_feats['target']

    #The depth of 25 was empirically found to perform well and be reasonably fast
    model = RandomForestClassifier(max_depth=25, random_state=0)

    model = model.fit(X,y)

    train_preds = model.predict(X)

    X_test = test_feats[[col for col in test_feats.columns if col != 'target']]
    y_test = test_feats['target']

    test_preds = model.predict(X_test)


    train_f1_scores = f1_score(y, train_preds, average=None)
    test_f1_scores = f1_score(y_test, test_preds, average=None)

    print(f"Depth {depth}")
    print(f'\nTrain f1 scores:\n{train_f1_scores}')
    print(f'Test f1 scores:\n{test_f1_scores}')
    

    
