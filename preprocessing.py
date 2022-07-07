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




def has_ambiguous_label(df):
    """Identify and return boolean filter of rows where two or more classes
    have identical number of votes (ie. there is no unique class 
    that has most votes).

    Args:
        df (pd.DataFrame): df with 'sampleid' and classes with votes and 'geometry'

    Returns:
        pd.DataFrame: contains rows with
    """
    df = df.set_index('sampleid')
    df['most_votes'] = df.max(axis=1)
    df['second_most_votes'] = df.drop(['most_votes', 'geometry'], axis=1).apply(lambda row: row.nlargest(2).values[-1],axis=1)

    return df.most_votes == df.second_most_votes



def drop_if_not_file_exists(dataframe, folder, name_column='sampleid'):

    paths = []
    for i, row in dataframe.iterrows():
        filename = f'{row[name_column]}.tif'
        paths.append(filename)

    dataframe['filename'] = paths

    drop_indices = []
    for index, row in dataframe.iterrows():
        if not os.path.isfile(os.path.join(folder, row.filename)):
            drop_indices.append(index)

    dataframe = dataframe.drop(drop_indices)

    return dataframe




if __name__ == '__main__':


    #controls = pd.read_csv('data/ILUC_controls_labels.csv')
    campaign = pd.read_csv('data/ILUC_campaign_labels.csv')


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
