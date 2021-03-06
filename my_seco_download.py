'''
This script downloads images according to given parameters and creates 
a median from them.



Example of use:

python my_seco_download.py  --save_path data/seco_campaign_landsat\
    --sensor L8\
    --sample_points_path data/campaign_locations_with_most_voted_class.shp\
    --num_workers 32

'''


import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from collections import OrderedDict
import time
from datetime import datetime, timedelta, date
import warnings
warnings.simplefilter('ignore', UserWarning)
import pdb
from dateutil.relativedelta import relativedelta

import pandas as pd
import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine
from skimage.exposure import rescale_intensity
import geopandas as gpd

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
BANDS_LANDSAT = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
BANDS_10M_S2 = ['B2', 'B3', 'B4', 'B8']

LAST_IDX_LOGFILE = "last_idx.txt"


class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()


class UniformSampler(GeoSampler):

    def sample_point(self):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]



def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask)




class ShapeFileSampler(GeoSampler):

    def __init__(self, shapefile_path):

        self.points = gpd.read_file(shapefile_path)

        self.start_index = 0
        self.end_index = self.points.index[-1]
        
        if os.path.exists(LAST_IDX_LOGFILE):
            self.index = int(open(LAST_IDX_LOGFILE).read())
            self.actual_started_index = self.index
            print(f"starting from idx {self.index} from {LAST_IDX_LOGFILE}")
        else:
            self.index = self.points.index[0]
            self.actual_started_index = self.points.index[0]


    def sample_point(self):
        if self.index > self.actual_started_index:
            with open(LAST_IDX_LOGFILE, 'w') as f:
                f.write(str(self.index))
        point = self.points.iloc[self.index]
        sampleid = point.sampleid
        geom = point.geometry
        lon, lat = geom.x, geom.y

        if self.index % 100 == 0:
            elapsed_seconds = (time.time() - start_time)
            eta = elapsed_seconds / (self.index - self.actual_started_index + 1) * (self.end_index - self.start_index)
            print("index:", self.index, 'elapsed_time:', str(timedelta(seconds=elapsed_seconds)), 'estimated_remaining_time:',  str(timedelta(seconds=eta-elapsed_seconds)))
        self.index += 1
        return [lon, lat], sampleid



def get_collection(sensor, cloud_pct):
    
    if sensor == 'S2':
        from gee_s2_cloudless_seco import S2cloud_processor
        #TODO: there are many parameters
        processor = S2cloud_processor()
    else:
        from gee_landsat_cloudless_seco import Landsat_cloud_processor
        processor = Landsat_cloud_processor(landsat_version=sensor[-1], max_cloud=cloud_pct)

    collection = processor.get_cloudfree_coll()
    return collection

def filter_collection(collection, coords, period=None):
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region
    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_patch(image, coords, sensor, radius, bands=None):

    region = ee.Geometry.Point(coords).buffer(radius).bounds()

    if sensor == 'S2':
        # From https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR:
        # The assets contain 12 UINT16 spectral bands representing SR scaled by 10000 (unlike in L1 data, there is no B10)
        scaled_image = image.divide(10000)
    else:
        #For landsat: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
        scaled_image = image.multiply(0.0000275).add(-0.2)
    
    patch = scaled_image.select(*bands).sampleRectangle(region, defaultValue=0)

    features = patch.getInfo()  # the actual download
    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features['properties'][band])
        img = rescale_intensity(img, in_range=(0, 0.3), out_range=np.uint8)
        raster[band] = img

    coords = np.array(features['geometry']['coordinates'][0])
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]

    patch = OrderedDict({
        'raster': raster,
        'coords': coords,
        'metadata': get_properties(image)
    })

    return patch


def get_patches_for_period(collection, coords, sensor, radius, bands=None):
    if bands is None:
        bands = RGB_BANDS

    max_images = 10

    if sensor == 'S2':
        cloud_attr = 'CLOUDY_PIXEL_PERCENTAGE'
    else:
        cloud_attr = 'CLOUD_COVER'

    collection = collection.limit(max_images, cloud_attr)
    collection_size = collection.size().getInfo()
    images = collection.toList(collection_size)

    patches = [get_patch(ee.Image(images.get(i)), coords, sensor, radius, bands) for i in range(collection_size)]

    return patches



def date2str(date):
    return date.strftime('%Y-%m-%d')


def get_patches_for_location(collection, coords, periods, sensor, debug=False, **kwargs):
    
    filtered_collections = [filter_collection(collection, coords, p) for p in periods]
    patches = [get_patches_for_period(coll, coords, sensor, **kwargs) for coll in filtered_collections]

    return patches



def save_geotiff(img, coords, filename):
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': channels,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'lzw'
    }

    with rasterio.open(filename, 'w', **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(raster, coords, metadata, path):
    
    patch_id = metadata['system:index']

    try:
        img = np.concatenate([v for k, v in raster.items()],axis=2)
    #When saving median, it's already a np array
    except AttributeError:
        img = raster

    save_geotiff(img, coords, os.path.join(path, f'{patch_id}.tif'))


def filter_patches_by_size(patches):
    """If list of patches contains patches of different size, it 
    cannot be concatenated. Here, given a list of patches, the most
    common shape is found and a list of patches of that size only
    is returned.

    Args:
        patches (list)

    Returns:
        list: filtered patches
    """
    #Get the name of first band (it is assumed all bands have the same shape)
    band_name = next(iter(patches[0]['raster'].items()))[0]
    #Find the most common shape
    shapes = [patch['raster'][band_name].shape for patch in patches]
    most_common = max(set(shapes), key=shapes.count)
    #Keep only those patches with the most common shape
    filtered_patches = [patch for patch in patches if patch['raster'][band_name].shape == most_common]

    return filtered_patches

def get_time_periods(start_date, end_date, period):
    """Generate a list of periods (each period is a tuple of dates).

    Args:
        start_date (str): in the form of YYYY-MM-DD
        end_date (str):  in the form of YYYY-MM-DD
        period (int): length of one period in months

    Returns:
        list: list of tuples (start_date, end_date) for each period
    """
    start = date(*[int(i) for i in start_date.split('-')]) 
    end = date(*[int(i) for i in end_date.split('-')])
    
    period_length_months = int(period)

    periods = []

    while start < end:
        periods.append((date2str(start), date2str(start+ relativedelta(months=+period_length_months))))
        start += relativedelta(months=+period_length_months)

    return periods


def compute_median(patches):
    """Compute median from provided images; If images have different 
    size, only compute median from a subset of images that have the same
    dimension.

    Args:
        patches (list): list of dicts; each dict['raster'] is a np.array of W*H*C

    Returns:
        np.array: nanmedian of patches; shape W*H*C
    """
    try:
        all_patches = np.array([np.concatenate([v for k, v in patch['raster'].items()],axis=2) for patch in patches]).astype('float')
    except ValueError: #Some patches have different shapes and therefore cannot be concatenated
        filtered_patches = filter_patches_by_size(patches)
        #Use only the filtered patches that agree in size
        all_patches = np.array([np.concatenate([v for k, v in patch['raster'].items()],axis=2) for patch in filtered_patches]).astype('float')
    
    all_patches[np.all(all_patches == 0, axis = 3)] = np.nan
    median = np.rint(np.nanmedian(all_patches, axis=0)).astype('uint8')    

    return median


class Counter:

    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_points_path', type=str)
   
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_path', type=str,)

    parser.add_argument('--cloud_pct', type=int, default=50)
    parser.add_argument('--sensor', type=str)
    parser.add_argument('--buffer', type=int, default=1000)

    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default='2021-01-01')
    parser.add_argument('--period_length_months', type=int, default=12)

    args = parser.parse_args()

    ee.Initialize()

    sensor = args.sensor
    buffer = args.buffer    
    assert sensor in ['S2', 'L7', 'L8'], f"Implemented sensors are 'S2', 'L7', 'L8', not {sensor}"
    collection = get_collection(sensor=sensor, cloud_pct=args.cloud_pct)

    sampler = ShapeFileSampler(args.sample_points_path)
    start_index = sampler.index
    end_index = sampler.points.index[-1]

    if sensor.startswith('L'):
        bands = BANDS_LANDSAT
    else:
        bands = BANDS_10M_S2

    time_periods = get_time_periods(args.start_date, args.end_date, args.period_length_months)

    out_folder_median = os.path.join(args.save_path, 'medians')
    os.makedirs(out_folder_median, exist_ok=True)

    start_time = time.time()
    counter = Counter()

    def worker(idx):
        try:
            coords, sampleid = sampler.sample_point()
            patches = get_patches_for_location(collection, coords, time_periods, sensor, radius=buffer, bands=bands, debug=args.debug)
            #sampleid = sampler.points.iloc[idx].sampleid
            location_path = os.path.join(args.save_path, f'{sampleid}')
            os.makedirs(location_path, exist_ok=True)
            for patches_in_period, period in zip(patches, time_periods):
                period_str = period[0] + '_' + period[1]
                out_folder = os.path.join(location_path, period_str)
                os.makedirs(out_folder, exist_ok=True)
                for patch in patches_in_period:
                    save_patch(
                        raster=patch['raster'],
                        coords=patch['coords'],
                        metadata=patch['metadata'],
                        path=out_folder,
                    )
                                
                median = compute_median(patches_in_period)

                save_patch(
                    raster=median,
                    coords=patch['coords'],
                    metadata={'system:index': f'{sampleid}_{period_str}'},
                    path=out_folder_median,
                )

            count = counter.update(len(patches))
            if count % args.log_freq == 0:
                print(f'Downloaded {count} images in {time.time() - start_time:.3f}s.')

        except Exception as e:
            print(f'Failed for sampleid {sampleid} with error:')
            print(e)


    indices = range(start_index, end_index)

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
