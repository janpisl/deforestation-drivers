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


#controls = pd.read_csv('data/ILUC_controls_labels.csv')
campaign = pd.read_csv('data/ILUC_campaign_labels.csv')

folder = 'data/campaign_L8_examples'


gdf = gpd.read_file('data/geowiki/ILUC.shp')

campaign = campaign.drop(['Unnamed: 0'], axis=1)
campaign = campaign.set_index('sampleid')
pdb.set_trace()

# I add geometry to the DataFrame with labels.
indexed_geoms = gdf[['sampleid', 'geometry']].drop_duplicates().set_index('sampleid')
filtered_geoms = indexed_geoms.loc[campaign.index]
campaing_processed = pd.concat([campaign, filtered_geoms], axis=1)


paths = []
for i, row in campaing_processed.iterrows():
    filename = f'{i}_2020_1.tif'
    paths.append(filename)

campaing_processed['filename'] = paths

drop_indices = []
for index, row in campaing_processed.iterrows():
    if not os.path.isfile(os.path.join(folder, row.filename)):
        drop_indices.append(index)

campaing_processed = campaing_processed.drop(drop_indices)

