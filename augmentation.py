import os
import pdb
import random
import glob
from itertools import combinations

import rasterio
import pandas as pd
import geopandas as gpd



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
    #Find the most common shape
    shapes = [patch.shape for patch in patches]
    most_common = max(set(shapes), key=shapes.count)
    #Keep only those patches with the most common shape
    filtered_patches = [patch for patch in patches if patch.shape == most_common]

    return filtered_patches



def compute_median(patches):
    """Compute median from provided images; If images have different 
    size, only compute median from a subset of images that have the same
    dimension.

    Args:
        patches (list):  each item is a np.array of W*H*C

    Returns:
        np.array: nanmedian of patches; shape W*H*C
    """
    try:
        all_patches = np.array([patch for patch in patches]).astype('float')
    except ValueError: #Some patches have different shapes and therefore cannot be concatenated
        filtered_patches = filter_patches_by_size(patches)
        #Use only the filtered patches that agree in size
        all_patches = np.array([patch for patch in filtered_patches]).astype('float')
    
    all_patches[np.all(all_patches == 0, axis = 3)] = np.nan
    median = np.rint(np.nanmedian(all_patches, axis=0)).astype('uint8')    

    return median



def compute_mean(patches):
    """Compute mean from provided images; If images have different 
    size, only compute mean from a subset of images that have the same
    dimension.

    Args:
        patches (list):  each item is a np.array of W*H*C

    Returns:
        np.array: nanmean of patches; shape W*H*C
    """
    try:
        all_patches = np.array([patch for patch in patches]).astype('float')
    except ValueError: #Some patches have different shapes and therefore cannot be concatenated
        filtered_patches = filter_patches_by_size(patches)
        #Use only the filtered patches that agree in size
        all_patches = np.array([patch for patch in filtered_patches]).astype('float')
    
    all_patches[np.all(all_patches == 0, axis = 3)] = np.nan
    mean = np.rint(np.nanmean(all_patches, axis=0)).astype('uint8')    

    return mean


def write_mean_median(data, profile, out_path):

    median = compute_median(data)
    mean = compute_mean(data)


    with rasterio.open(out_path.replace(".tif", "_median.tif"), 'w', **profile) as f:
        f.write(median)

    with rasterio.open(out_path.replace(".tif", "_mean.tif"), 'w', **profile) as f:
        f.write(mean)


def get_tif_data(paths):
    
    all_data = []
    for image_path in paths:
        with rasterio.open(image_path) as src:
            data = src.read()
            profile = src.profile
        all_data.append(data)

    return all_data, profile

def sample_sets(_list, set_length, n_sets):
    
    all_permutations = [i for i in combinations(_list, r=set_length)]

    return random.sample(all_permutations, n_sets)



if __name__ == '__main__':

    path = 'data/seco/L8/campaign/2020/annual'
    parent_out_dir = os.path.join(path, 'means_medians')

    folders = glob.glob(path + '/*')

    

    min_images = 5
    n_samples = 5

    for folder in folders:
        assert len(os.listdir(folder)) == 1, f"Expected 1 folder in {folder}, got these: {os.listdir(folder)}"
        
        images = glob.glob(folder + "/2020-01-01_2021-01-01/*.tif")        

        sampleid = folder.split("/")[-1]

        pdb.set_trace()

        max_images = len(images) - 1

        for n_images in range(min_images,max_images + 1):
            sets = sample_sets(images, n_images, n_samples)
            out_folder = os.path.join(parent_out_dir, f'{sampleid}')
            os.makedirs(out_folder, exist_ok=True)
            for i, set in enumerate(sets):
                all_data, profile = get_tif_data(set)
                write_mean_median(all_data, profile, os.path.join(out_folder, f'{sampleid}_images_{n_images}_set_{i}.tif'))