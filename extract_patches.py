import os
import pdb

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def get_squares(path, buffer=500):
    """For given dataset of points, extract all unique geometries,
     change geometry to squared buffers and return it as a GDF

    Args:
        path (str)

    Returns:
        GeoDataFrame:
    """
    df = pd.read_csv(path)


    location_gdf = gpd.GeoDataFrame({'geometry':gpd.points_from_xy(df.x, df.y), 'sampleid': df['sampleid']})
    unique_locations = location_gdf.drop_duplicates()

    unique_locations.crs = 4326
    #3857 projection has meter units (for buffering) and should work ok around the equator
    controls_gdf_proj = unique_locations.to_crs(3857)
    squares = controls_gdf_proj.geometry.buffer(buffer, cap_style=3).to_crs(4326)

    unique_locations['geometry'] = squares

    return unique_locations




if __name__ == '__main__':

    images_folder = 'data/controls_grid_05_L8_maxcloud_50_2020_FIXED_BANDS'
    controls_path = 'data/geowiki/ILUC_DARE_x_y/ILUC_DARE_controls_x_y.csv'

    out_folder = '/Users/janpisl/Documents/EPFL/drivers/data/controls_L8_examples'

    files = os.listdir(images_folder)

    grid = gpd.read_file('data/controls_grid_0_5.shp')
    squares = get_squares(controls_path, 560)
    for i, grid_cell in grid.iterrows():

        locations_in_grid_cell =  squares.loc[squares.geometry.intersects(grid_cell.geometry)]

        assert len(locations_in_grid_cell) > 0, "Empty grid cell, index {i}"
        
        #TODO: add the same with _2020_2.tif
        filename = f'idx_{i}_year_2020_1.tif'
        file_path = os.path.join(images_folder, filename)

        try:

            with rasterio.open(file_path) as source:
                
                for j, location in locations_in_grid_cell.iterrows():

                    out_image, out_transform = rasterio.mask.mask(source, [location.geometry], crop=True, all_touched=True)
                    out_meta = source.meta
                    if np.isnan(out_image).any():
                        continue
                    out_meta.update({"driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform})                 

                    out_path = os.path.join(out_folder,f"{location.sampleid}_2020_1.tif" )

                    with rasterio.open(out_path, "w", **out_meta) as sink:
                        sink.write(out_image)

            
        except rasterio.errors.RasterioIOError:
            print(f'File {filename} missing')

