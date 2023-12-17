import rasterio as rio
import numpy as np
import cv2
import glob
import tifffile
import os
import geopandas as gpd
import glob
from tqdm import tqdm
import detectree as dtr
import pickle
import pandas as pd
import subprocess
from  rasterio.features import geometry_mask
from shapely.geometry import mapping
import warnings
import json

# rasterio
import rasterio.control
import rasterio.crs
import rasterio.sample
import rasterio.vrt
import rasterio._features

with open('config.json', 'r') as f:
    c = json.load(f)

CONST_DIR_INNER_DATA = 'iner_data\\'

def create_masks(root_dir = '', images_dir = c['path']['train_tiles_dir'], masks_dir = c['path']["masks_dir"]):
    # List to store matching filenames of tiles and masks
    matching_files = []

    # Iterate through tiles and masks, match filenames, and store in matching_files list
    for tile_path in glob.glob(root_dir + images_dir + '*tif'):
        for mask_path in glob.glob(root_dir + masks_dir + '*tif'):
            if os.path.basename(tile_path) == os.path.basename(mask_path):
                matching_files.append(os.path.basename(tile_path))

    # Process tiles and masks
    for tile_path in glob.glob(root_dir + images_dir + '*tif'):
        # Check if the tile doesn't have a corresponding mask
        if os.path.basename(tile_path) not in matching_files:
            # Create a black mask
            img = np.zeros(cv2.imread(root_dir + images_dir + os.path.basename(tile_path))[:,:,0].shape)
            data = np.dstack((img, img, img)).astype('uint8')

            # Save the mask in RGB format
            tifffile.imwrite(
                root_dir + masks_dir + os.path.basename(tile_path),
                data,
                photometric='rgb',
                bitspersample=8,
                tile=(32, 32),
                planarconfig=1
            )

            # Preserve geospatial information from the original tile
            with rio.open(tile_path) as src:
                source_transform = src.transform
                source_crs = src.crs

            # Open the mask and preserve its geospatial information
            with rio.open(root_dir + masks_dir + os.path.basename(tile_path), 'r+') as dst:
                dst.transform = source_transform
                dst.crs = source_crs
        else:
            # Load the mask and convert it to binary (0 and 255)
            img = cv2.imread(root_dir + masks_dir + os.path.basename(tile_path))[:,:,0]
            img = np.where(img != 0, 255, 0)

            # Save the mask in RGB format
            data = np.dstack((img, img, img)).astype('uint8')
            tifffile.imwrite(
                root_dir + masks_dir + os.path.basename(tile_path),
                data,
                photometric='rgb',
                bitspersample=8,
                tile=(32, 32),
                planarconfig=1
            )

            # Preserve geospatial information from the original tile
            with rio.open(tile_path) as src:
                source_transform = src.transform
                source_crs = src.crs

            # Open the mask and preserve its geospatial information
            with rio.open(root_dir + masks_dir + os.path.basename(tile_path), 'r+') as dst:
                dst.transform = source_transform
                dst.crs = source_crs

    
if (c['model']['train']):    
    create_masks(root_dir = '', images_dir = c['path']['train_tiles_dir'], masks_dir = c['path']["masks_dir"])

    """**Start this code and grab a beer it is going to take a while...**

    the model will be trained and saved for future use
    """

    with open(CONST_DIR_INNER_DATA + 'split_df.pkl', 'rb') as f:
        split_df = pickle.load(f)

    # train a tree/non-tree pixel classfier
    clf = dtr.ClassifierTrainer().train_classifier(
        split_df= split_df, response_img_dir= c['path']["masks_dir"])

    #save the model in case of a crash or something or future use
    with open(CONST_DIR_INNER_DATA + 'tree_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

