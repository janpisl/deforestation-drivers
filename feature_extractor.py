from scipy import stats
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops

import skimage
from skimage.measure import shannon_entropy
from skimage.exposure import rescale_intensity
import torch



def feature_extractor(dataset, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM and other features from a dataset.

    Arguments:
        dataset -- numpy array
        distances -- list of ints
        angles -- list of doubles

    Returns:
        Pandas DataFrame
    """
    band_ranges = [
        (-3.0746,34.1025),
        (-3.4002, 28.9216),
        (-4.6208, 18.7967),
        (-2.9438, 13.1285),
        (-9.5916, 1.3183),
        (-5.0565, 2.9012),
        (-3.2917, 6.1823)
         ]

    

    all_features = []

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        

        bands = img.shape[0]      

        features = {}
        for b in range(bands):
            band = img[:,:,b]
            band = rescale_intensity(band, in_range=band_ranges[b], out_range=np.uint8) 
            band_features = get_features_for_band(band, b, distances, angles)
            features = {**features, **band_features}
            features['target'] = torch.argmax(label).item()

        all_features.append(features)
        

    all_features = pd.DataFrame(all_features)
        
    return all_features



def get_haralick_features(band, band_name, 
                          dist, angle, 
                          corr=True, 
                          homog=True, 
                          energy=False, 
                          diss=False, 
                          contrast=False):
    """Extract haralick features from a band.
 
    Arguments:
        band -- numpy array
        band_name -- str
        dist -- int
        angle -- float
        corr, homog, energy, diss, contrast -- bool
    Returns:
        dict       
    """

    df = {}
    angle_label = get_angle_label(angle)
    GLCM = graycomatrix(band, [dist], [angle])    

    if energy:  
        df[f'Band_{band_name}_Energy_{dist}_{angle_label}'] = graycoprops(GLCM, 'energy')[0][0]
    if corr:
        df[f'Band_{band_name}_Corr_{dist}_{angle_label}'] = graycoprops(GLCM, 'correlation')[0][0]   
    if diss:
        df[f'Band_{band_name}_Diss_sim_{dist}_{angle_label}'] = graycoprops(GLCM, 'dissimilarity')[0][0]
    if homog:
        df[f'Band_{band_name}_Homogen_{dist}_{angle_label}'] = graycoprops(GLCM, 'homogeneity')[0][0]       
    if contrast:
        df[f'Band_{band_name}_Contrast_{dist}_{angle_label}'] = graycoprops(GLCM, 'contrast')[0][0]

    return df


def get_angle_label(angle):
    """Get label for angle.

    Arguments:
        angle -- float
    
    Returns:
        str
    """
    if angle == 0:
        label = "0"
    elif angle == np.pi/4:
        label = "0.25_pi"
    elif angle == np.pi/2:
        label = "0.5_pi"
    elif angle == 3*np.pi/4:
        label = "0.75_pi"
    else:
        raise ValueError(f"Unsupported angle value: {j}")

    return label



def get_stats(band,band_name, mean=True, variance=True, skewness=False, kurtosis=False):
    """Get selected statistics of a band.
    
    Arguments:
        band -- numpy array
        band_name -- str
        mean -- bool
        variance -- bool
        skewness -- bool
        kurtosis -- bool
    """
    
    df = {}

    s = stats.describe(band.ravel())
    if mean:
        df[f'Band_{band_name}_Mean'] = s.mean
    if variance:
        df[f'Band_{band_name}_Variance'] = s.variance
    if skewness:
        df[f'Band_{band_name}_Skewness'] = s.skewness
    if kurtosis:
        df[f'Band_{band_name}_Kurtosis'] = s.kurtosis

    return df


def get_features_for_band(band,b, distances, angles, glcms=True, statistics=True, entropy=False):
    '''Extract features for a given band.

    Arguments:
        band -- numpy array
        b -- str
        distances -- list of ints
        angles -- list of floats
        glcms, statistics, entropy -- bool
    Returns:
        dict
    '''

    features = {}
    if glcms:
        for i in distances:
            for j in angles:
                haralick_features = get_haralick_features(band, b, i, j)
                features = {**features, **haralick_features}
    if statistics:
        features = {**features, **get_stats(band,b)}

    if entropy:
        entropy = shannon_entropy(band)
        features[f'Band_{b}_Entropy'] = entropy
    
    return features