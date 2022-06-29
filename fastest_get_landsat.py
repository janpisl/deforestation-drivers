
'''

'''
import argparse
import ee
import geopandas as gpd
import pandas as pd
import ee
import geetools
from geetools import tools
import numpy as np

import pdb 

from shapely.geometry import Point, Polygon
from tqdm import tqdm

from gee_landsat_cloudless import Landsat_cloud_processor

# Trigger the authentication flow.
#ee.Authenticate()

# Initialize the library.
ee.Initialize()



def get_squares(path, buffer=500):
    """For given dataset of points, extract all unique geometries,
     change geometry to squared buffers and return it as a GDF

    Args:
        path (str)

    Returns:
        GeoDataFrame:
    """
    df = pd.read_csv(path)

    gdf = gpd.GeoDataFrame(df,
                        geometry=gpd.points_from_xy(df.x, df.y))    

    location_gdf = gpd.GeoDataFrame({'geometry':gdf['geometry']})
    unique_locations = location_gdf.drop_duplicates()

    unique_locations.crs = 4326
    #3857 projection has meter units (for buffering) and should work ok around the equator
    controls_gdf_proj = unique_locations.to_crs(3857)
    squares = controls_gdf_proj.geometry.buffer(buffer, cap_style=3).to_crs(4326)

    unique_locations['geometry'] = squares

    return unique_locations





def get_grid(xmin, ymin, xmax, ymax, length, width):
    """Generate a grid from given parameters

    Args:
        xmin,ymin,xmax,ymax (float): bounding box
        length, width (float): size of a cell

    Returns:
        GeoDataFrame: GDF containing the grid
    """

    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + length, length))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+width, y), (x+width, y+length), (x, y+length)]))

    grid = gpd.GeoDataFrame({'geometry':polygons})

    return grid




def generate_aois(grid_size, squares_gdf):

    xmin, ymin, xmax, ymax = squares_gdf.total_bounds

    grid = get_grid(xmin, ymin, xmax, ymax, 5, 5)
    grid_containing_points = grid.loc[grid.apply(lambda row: squares_gdf.intersects(row.geometry).any(), axis=1)]
    grid_containing_points.crs = 4326

    all_gdfs = []

    print('Generating AOIs')
    for i, row in grid_containing_points.iterrows():
        xmin, ymin, xmax, ymax = row.geometry.bounds
        small_grid = get_grid(xmin, ymin, xmax, ymax, grid_size, grid_size)
        small_grid.crs = 4326
        small_grid_containing_points = small_grid.loc[small_grid.apply(lambda _row: squares_gdf.intersects(_row.geometry).any(), axis=1)].copy()
        
        clipped_geoms = []
        for j, small_grid_cell in small_grid_containing_points.iterrows():

            points_within_grid_square_cell =  squares_gdf.loc[squares_gdf.geometry.intersects(small_grid_cell.geometry)]
            min_lon, min_lat, max_lon, max_lat = points_within_grid_square_cell.total_bounds
            polygon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat, ), (max_lon, min_lat )])

            clipped_geoms.append(polygon)

        small_grid_containing_points['geometry'] = clipped_geoms

        all_gdfs.append(small_grid_containing_points)

    merged_gdf = pd.concat(all_gdfs)

    return merged_gdf




def main(path,landsat_version, max_cloud,  year_start, year_end, images_yearly, aois_path=None, grid_size=None, buffer=None, ):
    name_pattern = '{name}'

    squares_gdf = get_squares(path, buffer)

    if not aois_path:
        assert grid_size is not None, "Must provide either the AOIs or size of grid to generate."
        geometries = generate_aois(grid_size, squares_gdf)
        geometries = geometries.reset_index()
        pdb.set_trace()
        geometries.to_file('data/controls_grid_0_5.shp')
    else:
        geometries = gpd.read_file(aois_path)


    for i, row in geometries.iterrows():
        #if i < 5910:
        #    continue
        if i != 4739:
            continue
        one_grid_square = row.geometry


        locations_in_cell = squares_gdf.loc[squares_gdf.geometry.intersects(one_grid_square)]
        locations_ee = ee.FeatureCollection(locations_in_cell.__geo_interface__)
        
        #Create a GEE geometry from selected polygon
        geojson = gpd.GeoSeries([one_grid_square]).__geo_interface__
        AOI = ee.Geometry.Polygon(geojson['features'][0]['geometry']['coordinates'], proj="epsg:4326", geodesic=False)

        images = []
        for year in range(year_start, year_end):
            
            for j in 1,2:

                if images_yearly == 2:
                    if j == 1:
                        START_DATE = ee.Date(f"{year}-01-01")
                        END_DATE = ee.Date(f"{year}-06-30")
                    elif j == 2:
                        START_DATE = ee.Date(f"{year}-07-01")
                        END_DATE = ee.Date(f"{year}-12-31")        
                elif images_yearly == 1:

                    if j == 1:
                        START_DATE = ee.Date(f"{year}-01-01")
                        END_DATE = ee.Date(f"{year}-12-31")
                    elif j == 2:
                        continue
                else:
                    raise NotImplementedError("Only 1 or 2 images yearly implemented at the moment.")

                processor = Landsat_cloud_processor()
                collection = processor.get_cloudfree_coll(START_DATE, END_DATE, locations_ee, landsat_version, AOI, max_cloud=max_cloud)

                median = (collection.select(['B2', 'B3', 'B4', 'B5'])
                                        .median())

                median = median.set({'name': f'idx_{i}_year_{year}_{j}'})
                images.append(median)

            coll = ee.ImageCollection(images)

            tasks = geetools.batch.Export.imagecollection.toDrive(
                        collection=coll,
                        folder=f"new_campaign_grid_05_L{landsat_version}_maxcloud_{max_cloud}_{year}",
                        namePattern=name_pattern,
                        region=AOI, 
                        scale=30,
                        verbose=True,
                        maxPixels=int(1e13)
                    )





if __name__ == "__main__":


    parser = argparse.ArgumentParser()
 
    parser.add_argument("--data_path", "-d", type=str, help="Path to geowiki csv", required=True)

    parser.add_argument("--aoi_path", "-aoi", type=str, help="Path to AOIs to be requested.")
    
    parser.add_argument("--max_cloud", "-c", type=int, help="Images over this threshold will be filtered out")

    parser.add_argument("--landsat", "-l", type=int, help="Landsat version.")
    parser.add_argument("--start_year", "-s", type=int, help="Start year.")
    parser.add_argument("--end_year", "-e", type=int, help="End year (not included).")
    parser.add_argument("--buffer", "-b", type=int, help="Buffer (in meters) around geowiki points")
    parser.add_argument("--images_yearly", "-img", type=int, help="Either 1 or 2 images per year.")
    
    #parser.add_argument("--drive_directory", "-out", type=str, help="Prefix to folder in drive")


    args = parser.parse_args()


    #data_path = 'data/geowiki/ILUC_DARE_x_y/ILUC_DARE_campaign_x_y.csv'
    data_path = args.data_path

    aois_path = args.aoi_path

    landsat_version = args.landsat
    max_cloud = args.max_cloud

    #grid_size = 0.5

    year_start = args.start_year
    year_end = args.end_year

    buffer= args.buffer

    images_yearly = args.images_yearly


    main(data_path, landsat_version, max_cloud, 
        year_start, year_end, 
        images_yearly=images_yearly,
        aois_path=aois_path, 
        #grid_size=grid_size,
        buffer=buffer,
        )


