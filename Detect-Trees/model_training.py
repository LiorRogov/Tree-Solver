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
    
    true_tile_folder = os.path.join(root_dir, images_dir)
    true_mask_folder = os.path.join(root_dir, images_dir)

    masks_list = glob.glob(os.path.join(true_mask_folder, '*.tif'))
    images_list = glob.glob(os.path.join(true_tile_folder, '*.tif'))

    # Iterate through tiles and masks, match filenames, and store in matching_files list
    for tile_path in images_list:
        for mask_path in masks_list:
            if os.path.basename(tile_path) == os.path.basename(mask_path):
                matching_files.append(os.path.basename(tile_path))


    # Process tiles and masks
    for tile_path in images_list:
        # Check if the tile doesn't have a corresponding mask
        if os.path.basename(tile_path) not in matching_files:
            # Create a black mask
            true_tile = cv2.imread(tile_path)
            mask = np.zeros(true_tile.shape[:2], dtype= np.uint8)
            # = np.dstack((img, img, img)).astype('uint8')

        else:
            # Load the mask and convert it to binary (0 and 255)
            img = cv2.imread(os.path.join(masks_dir, os.path.basename(tile_path)))[:,:,0].astype(np.uint8)
            mask = np.where(img != 0, 255, 0)

    
        # Preserve geospatial information from the original tile
        with rio.open(tile_path) as src:
            # Read metadata of input raster
            profile = src.profile
        
        # Define the output raster file manually
        profile['driver'] = 'GTiff'
        profile['height'] , profile['width'] = mask.shape

        # Manually set any additional parameters if needed
        # For example, setting the data type to float32
        profile['dtype'] = np.uint8
        profile['count'] = 1

        # Specify the output file path
        output_path = os.path.join(masks_dir, os.path.basename(tile_path))

        # Create output raster file manually
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)


    
    
if (c['model']['train']):    
    create_masks(root_dir = '', images_dir = c['path']['train_tiles_dir'], masks_dir = c['path']["masks_dir"])

    """**Start this code and grab a beer it is going to take a while...**

    the model will be trained and saved for future use
    """

    with open(os.path.join(CONST_DIR_INNER_DATA, 'split_df.pkl'), 'rb') as f:
        split_df = pickle.load(f)

    # train a tree/non-tree pixel classfier
    clf = dtr.ClassifierTrainer().train_classifier(
        split_df= split_df, response_img_dir= c['path']["masks_dir"])

    #save the model in case of a crash or something or future use
    with open(os.path.join(CONST_DIR_INNER_DATA, 'tree_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)

