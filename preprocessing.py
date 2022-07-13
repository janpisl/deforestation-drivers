"""
The goal here is to produce annotations, given folder with images 
"""

import pdb

import pandas as pd
import geopandas as gpd
import rasterio
import os
import pandas as pd
import numpy as np
from shapely.geometry import box


def has_ambiguous_label(df):
    #TODO: re-do to 'has_majority_label'
    """Identify and return boolean filter of rows where two or more classes
    have identical number of votes (ie. there is no unique class 
    that has most votes).

    Args:
        df (pd.DataFrame): df with 'sampleid' and classes with votes and 'geometry'

    Returns:
        pd.Series: rows with two or more top voted answers are set to True
    """
    df = df.set_index('sampleid')
    df['most_votes'] = df.max(axis=1)
    df['second_most_votes'] = df.drop(['most_votes', 'geometry'], axis=1).apply(lambda row: row.nlargest(2).values[-1],axis=1)

    return df.most_votes == df.second_most_votes


def has_single_label(df):
    """Identify and return boolean filter of rows with a single class
    that has all the votes

    Args:
        df (pd.DataFrame): df with 'sampleid' and classes with votes and 'geometry'

    Returns:
        pd.Series: rows where all votes are for the same class are True
    """
    df = df.set_index('sampleid')
    df['second_most_votes'] = df.drop(['most_votes', 'geometry', 'filename'], axis=1, errors='ignore').apply(lambda row: row.nlargest(2).values[-1],axis=1)

    return df.reset_index().drop('sampleid', axis=1, errors='ignore').second_most_votes == 0



def get_file_names(dataframe, name_column):
    paths = []
    for i, row in dataframe.iterrows():
        filename = f'{row[name_column]}.tif'
        paths.append(filename)

    return paths


def file_exists(dataframe, folder, name_column='sampleid'):
    
    if not 'filename' in dataframe.columns:
        dataframe['filename'] = get_file_names(dataframe, name_column=name_column)

    return dataframe.filename.apply(lambda filename: os.path.isfile(os.path.join(folder, filename)))



def has_missing_data(dataframe, folder, name_column='sampleid'):
    """Return boolean Series where

    Args:
        df (pd.DataFrame): 
        folder (str): directory with images
    Returns:
        pd.Series: True value if file has 0s over all bands in any pixel
    """
    def missing_data_across_bands(file):
        with rasterio.open(file) as src:
            data = src.read()
            if np.any(data.sum(axis=0) == 0):
                return True
            return False

    if not 'filename' in dataframe.columns:
        dataframe['filename'] = get_file_names(dataframe, name_column=name_column)

    return dataframe.filename.apply(lambda filename: missing_data_across_bands(os.path.join(folder, filename)) )


def find_id_sindex(file_name, sindex, gdf):

    with rasterio.open(file_name) as src:
        bounds = src.bounds
        geom = box(*bounds)

    potential_indices = sindex.intersection(geom.bounds)
    potential_matches = gdf.iloc[potential_indices]

    return potential_matches.loc[potential_matches.intersects(geom)].sampleid.values[0]


def fix_naming(image_folder, output_folder, gdf_labels_path='data/campaign_labels_processed.shp'):
    """For each image in folder, get its extent, find which location it intersects with,
    get the sampleid for that location and rename the image to f'{sampleid}.tif'

    Args:
        folder (str): path to folder with images to be renamed
        gdf_labels_path (str): path to GDF with 'sampleid' and geometries. Defaults to 'data/campaign_labels_processed.shp'.
    """
    gdf_labels = gpd.read_file(gdf_labels_path)
    os.makedirs(output_folder, exist_ok=True)
    sindex = gdf_labels.sindex

    files = [os.path.join(image_folder,f) for f in os.listdir(image_folder) if f.endswith('.tif')]

    for file_path in files:

        correct_index = find_id_sindex(file_path, sindex, gdf_labels)
        new_name = os.path.join(output_folder, f'{correct_index}.tif')
        
        os.rename(file_path, new_name)









if __name__ == '__main__':


    #controls = pd.read_csv('data/ILUC_controls_labels.csv')
    '''campaign = pd.read_csv('data/ILUC_campaign_labels.csv')


    gdf = gpd.read_file('data/geowiki/ILUC.shp')

    try:
        campaign = campaign.drop(['Unnamed: 0'], axis=1)
    except KeyError:
        pass
    campaign = campaign.set_index('sampleid')

    # I add geometry to the DataFrame with labels.
    indexed_geoms = gdf[['sampleid', 'geometry']].drop_duplicates().set_index('sampleid')
    filtered_geoms = indexed_geoms.loc[campaign.index]
    campaing_processed = pd.concat([campaign, filtered_geoms], axis=1)

    #folder = 'data/campaign_L8_examples'
    #filtered = drop_if_not_file_exists(campaing_processed, folder)

    filtered = campaing_processed[~has_ambiguous_label(campaing_processed)]


    folder = 'data/tmp/medians'
    annotations = 'data/tmp/annotations_with_majority_class.csv'
    
    img_labels = pd.read_csv(annotations)
    x = drop_if_not_file_exists(img_labels, folder)
    y = drop_if_missing_data(x, folder)

    folder = 'data/seco_campaign_landsat/medians'
    out_folder = 'data/seco_campaign_landsat/medians_fixed_naming'
    fix_naming(folder, out_folder)

    labels = 'data/tmp/annotations_with_majority_class.csv'
    counts = get_class_counts(pd.read_csv(labels))'''



    folder = 'data/tmp/renamed_medians'
    annotations = 'data/tmp/annotations_with_majority_class.csv'
    
    img_labels = pd.read_csv(annotations)
    existing_files = img_labels.loc[file_exists(img_labels, folder)]
    x = has_missing_data(existing_files, folder)
    pdb.set_trace()
    print()