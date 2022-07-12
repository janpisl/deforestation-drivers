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

import numpy as np
import torch
from dataset import GeoWikiDataset
from pipeline_softmax import get_datasets
from feature_extractor import feature_extractor
from sklearn.ensemble import RandomForestClassifier

device = torch.device('cpu')




if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--image_folder', type=str)



    parser.add_argument('--drop_rows_with_missing_vals', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--single_label_only', action=argparse.BooleanOptionalAction)    

    args = parser.parse_args()

    device = torch.device('cpu')

    annotations_path, images_path, drop_missing_vals, single_label_only = \
        args.annotations_path, args.image_folder, args.drop_rows_with_missing_vals, args.single_label_only, 


    torch.manual_seed(420)

    train_dataset, test_dataset, _ = get_datasets(annotations_path, images_path, drop_missing_vals, single_label_only, device)
    test = [train_dataset[i] for i in range(1000)]
    
    #inputs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])

    try:
        train_feats = feature_extractor([train_dataset[i] for i in range(len(train_dataset))])
        test_feats = feature_extractor([test_dataset[i] for i in range(len(test_dataset))])
    except:
      pdb.set_trace()
      feats = feature_extractor(test)

    model = RandomForestClassifier(max_depth=10, random_state=0)

    X = train_feats[[col for col in train_feats.columns if col != 'target']]
    y = train_feats['target']

    model.fit(X,y)
    pdb.set_trace()
    print()

    X_test = [col for col in test_feats.columns if col != 'target']
    y_test = test_feats['target']

    #print (clf.score(training, training_labels))
    #print(clf.score(testing, testing_labels))



    '''
    1. feature extractor: for every image in train_dataset and test_dataset, extract features per band: mean,med,min,max,variance, haralick, etc.
    2. classify features
    '''
