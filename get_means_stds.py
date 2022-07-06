import os
import rasterio
import pdb
import numpy as np
import argparse



def get_means_stds(folder):

    bands = 7

    accum_means, accum_stds = np.zeros(7), np.zeros(7)
    count = 0
    for i, _file in enumerate(os.listdir(path)):

        with rasterio.open(os.path.join(path,_file)) as source:
            data = source.read()

            if np.isnan(data).any():
                continue
            
            count += 1
            accum_means = np.add(accum_means, data.mean(axis=(1,2)))
            accum_stds = np.add(accum_stds, data.std(axis=(1,2)))

    return accum_means/count, accum_stds/count

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=str)
   
    args = parser.parse_args()

    path = args.input_folder
    #path = '/Users/janpisl/Documents/EPFL/drivers/data/controls_L8_examples'

    means, stds = get_means_stds(path)

    print(means)
    print(stds)

