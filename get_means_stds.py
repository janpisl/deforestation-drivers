import os
import rasterio
import pdb
import numpy as np


if __name__ == '__main__':

    path = '/Users/janpisl/Documents/EPFL/drivers/data/controls_L8_examples'
    accum_means = np.array([0,0,0,0])
    accum_stds = np.array([0,0,0,0])
    count = 0
    for _file in os.listdir(path):
        
        with rasterio.open(os.path.join(path,_file)) as source:
            data = source.read()

            if np.isnan(data).any():
                continue
            
            count += 1
            accum_means = np.add(accum_means, data.mean(axis=(1,2)))
            accum_stds = np.add(accum_stds, data.std(axis=(1,2)))


    print(accum_means/count)
    print(accum_stds/count)

