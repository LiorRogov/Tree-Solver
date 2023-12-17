from ast import Name
import detectree as dtr
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import plot
import pickle
import glob
import os
import cv2
import numpy as np
import pandas as pd
import poisson_disc
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
import xml.etree.ElementTree as ET
import tifffile
import warnings

from rasterio.features import shapes
from shapely.geometry import shape

from tqdm import tqdm

import pickle
import multiprocessing

import json

with open('config.json', 'r') as f:
    c = json.load(f)

CONST_DIR_INNER_DATA = 'iner_data\\'
results = c['path']['tiles_results']


#read configuration from xml file
maximum_radius_for_single_tree = c['tree']['maximum_radius_for_single_tree']
spacial_resulution = c['raw_data']['desired_resolution'][0]

# Use the glob module to search for a file named 'tree_model.pkl' in the current directory
# Assuming there's only one such file, get its path
file_path = glob.glob(CONST_DIR_INNER_DATA + 'tree_model.pkl')[0]

# Open the 'tree_model.pkl' file in binary read mode
with open(file_path, "rb") as f:
    # Load (deserialize) the machine learning model from the file
    clf = pickle.load(f)

# At this point, 'clf' contains the loaded machine learning model from 'tree_model.pkl'


def saperate_single_multiple_trees(properties ,y_pred, labels):
    # Create a pandas DataFrame from the properties list
    df_area_labels = pd.DataFrame(properties)

    combined_trees = np.zeros(y_pred.shape) 

    #print (df_area_labels['area'])
    #finding single trees
    single_tees = df_area_labels[np.sqrt(df_area_labels['area'] * np.power(spacial_resulution, 2) / np.pi) < maximum_radius_for_single_tree]

    for label in single_tees['label']:
        component_mask = labels == label
        # Calculate centroid (center of mass)
        non_zero_coords = np.argwhere(component_mask)
        centroid_y = int(np.mean(non_zero_coords[:, 0]))
        centroid_x = int(np.mean(non_zero_coords[:, 1]))
        combined_trees[centroid_y, centroid_x] = 1

    
    multipule_trees = df_area_labels[~(np.sqrt(df_area_labels['area'] * np.power(spacial_resulution, 2) / np.pi) < maximum_radius_for_single_tree)]

    return (multipule_trees, single_tees, combined_trees)

def create_possion_noise(y_pred):
    # Get the dimensions of the input array
    dims2d = np.array(y_pred.shape)
    # Generate Poisson disk sampling points using Bridson's algorithm
    points_surf = poisson_disc.Bridson_sampling(dims=dims2d, radius= c['poisson_disc_sampling']['radius'], k=30, hypersphere_sample=poisson_disc.hypersphere_surface_sample)
    return points_surf

def get_pixels_point_location_as_pandas(tile_path, combined_trees):
    # Open the GeoTIFF image using rasterio
    with rio.open(tile_path) as dataset:
        # Read the pixel values from the first band (index 1)
        val = dataset.read(1)

        # Get the no data value from the dataset
        no_data = dataset.nodata

        # Create a list of Point geometries for pixels where combined_trees is 1
        geometry = [Point(dataset.xy(x, y)[0], dataset.xy(x, y)[1]) for x, y in np.ndindex(val.shape) if combined_trees[x, y] == 1]

        # Create a list of pixel values (data) for the same set of pixels
        v = [val[x, y] for x, y in np.ndindex(val.shape) if combined_trees[x, y] == 1]

        # Create a GeoDataFrame with geometry and data columns
        df = gpd.GeoDataFrame({'geometry': geometry, 'data': v})

        # Set the coordinate reference system (CRS) of the GeoDataFrame to match the dataset's CRS
        df.crs = dataset.crs

    # Return the geometries as a pandas Series
    return df['geometry']

def save_tree_object(shapefile, y_pred, tile_path):
    # Extract the base name of the tile without the file extension
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]
    # Create a directory to save the results if it doesn't exist
    if not os.path.exists(f'{results}/{tile_name}'):
        os.makedirs(f'{results}/{tile_name}')

    # Save the shapefile (tree locations) to a shapefile format
    shapefile.to_file(f'{results}/{tile_name}/trees.shp')

    # Create a binary mask where non-zero values indicate tree pixels
    mask = np.where(y_pred != 0, 255, 0)

    # Save the tree mask as a GeoTIFF file with specified metadata
    tifffile.imwrite(
        f'{results}/{tile_name}/tree_mask.tif', 
        np.dstack((mask, mask, mask)).astype('uint8'),
        photometric='rgb',
        compression='LZW',
        bitspersample=8,
        tile=(32, 32),
        planarconfig=1  # chunky format
    )

    # Open the source TIFF dataset
    source_tiff_dataset = rio.open(tile_path)

    # Set the destination TIFF path
    destination_tiff_path = f'{results}/{tile_name}/tree_mask.tif'

    # Open the destination TIFF file for read/write ("r+")
    with rio.open(destination_tiff_path, "r+") as src:
        # Copy geotransform and CRS from the source to the destination
        src.transform = source_tiff_dataset.transform
        src.crs = source_tiff_dataset.crs

def save_grass_object(y_pred, tile_path):
    # Extract the tile name without the file extension
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]

    # Create a binary mask where non-zero values indicate tree pixels
    mask = np.where(y_pred != 0, 1, 0)

    # Open the source GeoTIFF file and extract information
    with rio.open(tile_path) as src:
        img = np.dstack((src.read(1), src.read(2), src.read(3)))  # Read RGB channels
        transform = src.transform  # Get spatial transformation
        crs = src.crs  # Get CRS (Coordinate Reference System)

    # Convert the RGB image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

    # Define an HSV range for the color you want to detect (green color)
    lower_hsv = np.array(c['grass_color']['lower_hsv'])
    upper_hsv = np.array(c['grass_color']['upper_hsv'])

    # Create a mask to filter the color within the specified range
    color_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # Create a mask for detected trees by combining the tree mask and color mask
    trees_mask = mask
    detected_trees_mask = np.where((trees_mask == 0) & (color_mask != 0), 255, 0)

    # Apply morphological operations (opening) to clean up the mask
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(detected_trees_mask.astype('uint8'), cv2.MORPH_OPEN, kernel)

    # Save the resulting mask as a GeoTIFF
    tifffile.imwrite(
        f'{results}/{tile_name}/grass_mask.tif',
        np.dstack((opening, opening, opening)).astype('uint8'),
        photometric='rgb',
        compression='LZW',
        bitspersample=8,
        tile=(32, 32),
        planarconfig=1  # chunky format
    )

    # Open the source TIFF dataset
    source_tiff_dataset = rio.open(tile_path)

    # Set the destination TIFF path
    destination_tiff_path = f'{results}/{tile_name}/grass_mask.tif'

    # Open the destination TIFF file for read/write ("r+")
    with rio.open(destination_tiff_path, "r+") as src:
        # Copy geotransform and CRS from the source to the destination
        src.transform = source_tiff_dataset.transform
        src.crs = source_tiff_dataset.crs

    # Get the mask array
    mask_array = opening

    # Convert the mask to polygons
    shapes_ = shapes(mask_array, mask=mask_array > 0, transform=transform)
    polygons = [shape(geom) for geom, _ in shapes_]

    # Create a MultiPolygon from the polygons
    multipolygon = MultiPolygon(polygons)

    # Define the schema for the shapefile
    schema = {'geometry': 'MultiPolygon', 'properties': {}}

    # Create a GeoDataFrame from the MultiPolygon
    gdf = gpd.GeoDataFrame({'geometry': [multipolygon]}, crs=src.crs)

    # Specify the output shapefile path and save the GeoDataFrame to the shapefile
    output_shapefile = f'{results}/{tile_name}/grass.shp'
    gdf.to_file(output_shapefile, schema=schema)



def worker_function(target):
    # Ignore NotGeoreferencedWarning warnings from rasterio
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

    # Load data from 'tree_area_tiles.pkl'
    with open(CONST_DIR_INNER_DATA + 'tree_area_tiles.pkl', 'rb') as f:
        land_tiles = pickle.load(f)

    # Iterate over a specific range of tiles based on the 'target' argument
    for tile_path in tqdm(land_tiles[target[0]: target[1]]):
        tile_name = os.path.basename(tile_path)
        
        # Check if the 'results' directory exists for saving results
        if not os.path.exists(results + '\\' + os.path.splitext(tile_name)[0]):
            # Classify the image using a classifier (likely a machine learning model)
            y_pred = dtr.Classifier().classify_img(tile_path, clf)

            # Create a binary mask from the classification results
            binary_mask = np.where(y_pred == 0, 0, 1).astype('uint8')

            # Apply connected components labeling to identify tree components
            num_labels, labels = cv2.connectedComponents(binary_mask)

            # Calculate properties of each connected component (tree properties)
            properties = []
            for label in range(1, num_labels):
                component_mask = labels == label
                area = np.sum(component_mask)
                
                properties.append({
                    'label': label,
                    'area': area,
                })
            
            if len(properties) > 0:
                # Separate single trees from multiple trees based on properties
                multipule_trees, single_tees, combined_trees = saperate_single_multiple_trees(properties, y_pred, labels)

                # Create Poisson noise points (not shown in the code)
                points_surf = create_possion_noise(y_pred)

                # Iterate over multiple trees and add them to the combined_trees array
                for label in multipule_trees['label']:
                    component_mask = labels == label
                    for point in points_surf:
                        x = int(point[0])
                        y = int(point[1])
                        if x == component_mask.shape[0]:
                            x = int(point[0]) - 1
                        if y == component_mask.shape[1]:
                            x = int(point[1]) - 1
                        if component_mask[x, y] == 1:
                            combined_trees[x, y] = 1

                # Get pixel locations as a shapefile (not shown in the code)
                shapefile = get_pixels_point_location_as_pandas(tile_path, combined_trees)
                
                # Save tree object (not shown in the code)
                save_tree_object(shapefile, y_pred, tile_path)
                
                # Save grass object (not shown in the code)
                save_grass_object(y_pred, tile_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Get the number of CPU cores available on the system
    num_cores = multiprocessing.cpu_count()

    # Load data from 'tree_area_tiles.pkl'
    with open(CONST_DIR_INNER_DATA + 'tree_area_tiles.pkl', 'rb') as f:
        land_tiles = pickle.load(f)

    # Calculate the size of each sub-range of data for parallel processing
    divisor = len(land_tiles) // num_cores
    list_groups = []
    start = 0

    # Divide the data into sub-ranges based on the number of CPU cores
    for i in range(1, num_cores, 1):
        list_groups.append([start, divisor * i])
        start = divisor * i
    list_groups.append([start, len(land_tiles)])

    # Print the list of sub-ranges and the number of CPU cores
    print(list_groups)
    print(num_cores)

    # Create a list to store the process instances
    processes = []

    # Start a separate process for each sub-range of data
    for number in range(num_cores):
        process = multiprocessing.Process(target=worker_function, args=(list_groups[number],))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
