
'''

'''


import ee
import geopandas as gpd
import pandas as pd
import ee
import geetools
from geetools import tools

import pdb 

from shapely.geometry import Point
from tqdm import tqdm

from gee_s2cloudless import S2cloud_processor

# Trigger the authentication flow.
#ee.Authenticate()

# Initialize the library.
ee.Initialize()




####################
CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

name_pattern = '{name}'
####################




controls = pd.read_csv('data/ILUC_DARE_x_y/ILUC_DARE_controls_x_y.csv')

controls_gdf = gpd.GeoDataFrame(controls,
                     geometry=gpd.points_from_xy(controls.x, controls.y))    


controls_gdf.crs = 4326
#3857 projection has meter units (for buffering) and should work ok around the equator
controls_gdf_proj = controls_gdf.to_crs(3857)
squares = controls_gdf_proj.geometry.buffer(500, cap_style=3).to_crs(4326)

controls_gdf['geometry'] = squares

for i, row in tqdm(controls_gdf.loc[controls_gdf.step == 'step1'].iterrows()):
    if int(i) > 5:
        break
    bounds = row.geometry.bounds
    
    AOI = ee.Geometry.Polygon(
            [[[bounds[0],bounds[1]],
            [bounds[0],bounds[3]],
            [bounds[2],bounds[3]],
            [bounds[2],bounds[1]]]], proj="epsg:4326", geodesic=False)


    images_for_aoi = []

    for year in [2019,2020,2021]:

        START_DATE = ee.Date(f"{year}-01-01")
        END_DATE = ee.Date(f"{year}-12-31")

        processor = S2cloud_processor(AOI, START_DATE, 
                                      END_DATE,CLOUD_FILTER,
                                      CLD_PRB_THRESH,NIR_DRK_THRESH,
                                      CLD_PRJ_DIST,BUFFER)

        collection = processor.get_cloudfree_coll()

        s2_sr_median = (collection.select(['B2', 'B3', 'B4', 'B8'])
                                  .median())

        s2_sr_median = s2_sr_median.set({'name': f'test_{i}_{year}'})
        images_for_aoi.append(s2_sr_median)

    coll = ee.ImageCollection(images_for_aoi)
    
    tasks = geetools.batch.Export.imagecollection.toDrive(
                collection=coll,
                folder=f"s2_cloudless",
                namePattern=name_pattern,
                region=AOI,
                scale=10,
                verbose=True,
                maxPixels=int(1e13)
            )