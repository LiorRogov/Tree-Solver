"""
This section will explain how to extract the locations of trees from aerial imagery and create a shapefile containing points representing these tree locations.

importing neccesery libraries
"""
# rasterio
import rasterio.control
import rasterio.crs
import rasterio.sample
import rasterio.vrt
import rasterio._features

#the end for problematic libraries 
from sklearn import cluster, decomposition, metrics
import dask
import numpy as np
import cv2
import glob
import shutil
from affine import Affine
import os
import rasterio as rio
import geopandas as gpd
import glob
from tqdm import tqdm
import detectree as dtr
from detectree import TrainingSelector
import pickle
import pandas as pd
from shapely.geometry import Polygon
import json
from rasterio.features import Window
from rasterio.crs import CRS
from shapely.geometry import box

with open('config.json', 'r') as f:
        c = json.load(f)

def save_rescaled(tif_path,  output_dir, new_resolution=  tuple(c['raw_data']['desired_resolution']), current_resolution=tuple(c['raw_data']['current_resolution'])):

    """
    Rescale a GeoTIFF image while preserving georeferencing and save it as a new GeoTIFF.

    Parameters:
    - tif_path (str): The path to the input GeoTIFF file.
    - output_dir (str): The directory where the rescaled GeoTIFF file will be saved.
    - new_resolution (tuple, optional): The desired new resolution as a tuple of (y_resolution, x_resolution) in units per pixel. Default is (0.45, 0.45).
    - current_resolution (tuple, optional): The current resolution of the input GeoTIFF as a tuple of (y_resolution, x_resolution) in units per pixel. Default is (0.15, 0.15).

    Returns:
    - output_tiff (str): The path to the saved rescaled GeoTIFF.
    """

    # Calculate the ratio of the old and new resolutions along X and Y axes
    resolution_ratio_x = current_resolution[1] / new_resolution[1]
    resolution_ratio_y = current_resolution[0] / new_resolution[0]

    # Build the output file path for the rescaled GeoTIFF
    file_base_name, ext = os.path.splitext(os.path.basename(tif_path))
    output_tiff = os.path.join(output_dir, f'{file_base_name}_rescaled{ext}')

    # Open the original GeoTIFF file using rasterio and obtain metadata
    with rio.open(tif_path) as src:
        # Read image data and metadata
        transform = src.transform
        crs = src.crs

        # Create an Affine transformation matrix with the updated resolution
        new_transform = Affine(transform.a / resolution_ratio_x, transform.b, transform.c,
                        transform.d, transform.e / resolution_ratio_y, transform.f)

    # Read the image using OpenCV
    img = cv2.imread(tif_path)

    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate new dimensions based on resolution ratio
    new_width = int(img.shape[1] * resolution_ratio_x)
    new_height = int(img.shape[0] * resolution_ratio_y)

    # Perform the image rescaling using bilinear interpolation
    resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Define a new profile for the rescaled GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'width': new_width,
        'height': new_height,
        'count': 3,  # Set count to 3 for RGB image
        'crs': crs,
        'transform': new_transform
    }

    # Transpose the image data to match the expected shape (3, new_height, new_width)
    output_tile = np.transpose(resized, (2, 0, 1))

    # Create a new GeoTIFF file with the updated georeferencing and rescaled data
    with rio.open(output_tiff, 'w', **profile) as dst:
        dst.write(output_tile)

    # Return the path to the saved rescaled GeoTIFF
    return output_tiff

def to_tiles(rescled_path, output_dir, tile_dim):
    """
    Split a rescaled GeoTIFF image into smaller tiles and save them as separate GeoTIFF files.

    Parameters:
    - rescled_path (str): The path to the rescaled GeoTIFF image to be split into tiles.
    - output_dir (str): The directory where the individual tiles will be saved.
    - tile_dim (tuple, optional): The dimensions of each tile in pixels as (width, height). Default is (268, 365).
    """

    # Open the input GeoTIFF file
    with rio.open(rescled_path) as src:
        # Get metadata and profile information
        profile = src.profile
        transform = src.transform

        # Calculate tile dimensions
        tile_height = tile_dim[1]
        tile_width = tile_dim[0]

        # Get the shape (height and width) of the input GeoTIFF using OpenCV
        img_shape = cv2.imread(rescled_path).shape

        # Iterate through rows and columns to create tiles
        for row in range(img_shape[1] // tile_width):
            for col in range(img_shape[0] // tile_height):
                # Calculate the window for the current tile
                window = Window(row * tile_width, col * tile_height, tile_width, tile_height)

                # Read the data for the current tile
                tile_data = src.read(window=window)

                # Calculate the transform for the current tile
                tile_transform = src.window_transform(window)

                # Create a new profile for the current tile
                tile_profile = src.profile.copy()
                tile_profile.update(width=tile_width, height=tile_height, transform=tile_transform)

                # Generate a file path for the current tile
                tile_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(rescled_path))[0]}_{row}_{col}.tif")

                # Write the tile data as a new GeoTIFF with its own georeferencing
                with rio.open(tile_path, 'w', **tile_profile) as dst:
                    dst.write(tile_data)

    #print("Tiles created successfully.")  # Print a message indicating successful tile creation



def is_tile_intersects_with_object(tile_path ,polylines):
    # Open the GeoTIFF image specified by 'tile_path' using 'rasterio'
    with rasterio.open(tile_path) as src:
         # Get the georeferenced transform
        transform = src.transform

        # Get the image width and height
        width = src.width
        height = src.height

        # Calculate the coordinates of the four corners of the image
        image_polygon = box(*src.bounds)

        # Iterate through each polyline in the 'polylines' DataFrame
        for row in polylines.itertuples():
            # Get the LineString geometry from the row
            polyline = row[-1]

            # Check if the tile intersects with the polyline
            if polyline.intersects(image_polygon):
                return True

        # If there are no intersections with any polyline, return False
        return False

def is_tile_within_object(tile_path, polygons):
    # Initialize a counter to keep track of polygon intersections
    counter = 0

    # Open the GeoTIFF file specified by 'tile_path' using 'rasterio'
    with rasterio.open(tile_path) as src:
         # Get the georeferenced transform
        transform = src.transform

        # Get the image width and height
        width = src.width
        height = src.height

        image_polygon = box(*src.bounds)

        # Iterate through each polygon in the 'polygons' DataFrame
        for row in polygons.itertuples():
            # Get the LineString geometry from the row
            polygon = row[-1]

            # Check if the tile is entirely within the polygon
            if image_polygon.within(polygon):
                return True
            # Check if the tile intersects with the polygon
            elif image_polygon.intersects(polygon):
                counter += 1

    # If the counter is greater than 1, there is at least one intersection
    if counter > 1:
        return True

    # If there are no intersections, return False
    return False



if __name__ == "__main__":
    CONST_DIR_INNER_DATA = 'iner_data\\'
    os.makedirs(CONST_DIR_INNER_DATA, exist_ok= True)

    """**Data Preparation**"""
    
    """
    **Reduction aerial imagery resolution:
    Certainly, the level of detail and accuracy in aerial imagery is directly linked to its spatial resolution. A resolution of 0.5 meters per pixel is generally sufficient for tree extraction, but superior results come with higher spatial resolutions. However, higher resolution implies longer training times for tree detection models, creating a trade-off between accuracy and processing speed. Moreover, the processing time is not just determined by resolution but also by the volume of available aerial imagery. More imagery results in longer processing times. To manage this, reducing spatial resolution minimizes the raw data. All the above must be carefully considered based on the specific project requirements.
    """

    # Specify the path to the directory containing the original GeoTIFF files
    TRUE_IMAGERY_DIR = c['path']['raw_data']
    # Specify the directory where you want to save the new rescaled GeoTIFF files
    OUTPUT_DIR = c['path']['raw_data_rescaled']
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print ("raw tiff images rescaling has begun")
    # Iterate through all TIFF files in the TRUE_IMAGERY_DIR directory
    tiff_images_list = glob.glob(os.path.join(TRUE_IMAGERY_DIR, '*.tif'))
    if not (tiff_images_list):
        raise Exception('raw data folder is empty')
    
    for tif_image_path in tqdm(tiff_images_list):
        # Call the save_rescaled function to rescale and save each TIFF file
        path = save_rescaled(tif_image_path, OUTPUT_DIR)

    
    # Define the directory where the rescaled images are saved
    RAW_DATA_RESCALED_FOLDER = c['path']['raw_data_rescaled']
    
    # Define the directory where you want to save the tiles
    TILES_OUTPUT_FOLDER = c['path']['output_dir_for_tiles']
    os.makedirs(TILES_OUTPUT_FOLDER, exist_ok=True)
    
    print ('raw data rescaled to tiles, has been started')
    # Iterate through all TIFF files in the 'data' directory
    rescaed_imagery_filepaths_list = glob.glob(os.path.join(RAW_DATA_RESCALED_FOLDER, '*_rescaled.tif'))
    for tif_image_path in tqdm(rescaed_imagery_filepaths_list):
        # Call the 'to_tiles' function to split each rescaled image into tiles remember all the tiles must be at the same size. Therefore the Geotiff's resolution in data directory must be devisible by the tile's resulution.
        to_tiles(tif_image_path, TILES_OUTPUT_FOLDER, c['tile']['dim'])

    """**To Tiles:**
    After reducing the spatial resolution, the resized data needs to be divided into tiles, each with a resolution not exceeding 1000x1000. This is necessary for the Detectree library, which selects 1% of these tiles to effectively represent the entire dataset for labeling and training the model (explained in detail later).  
    """

    # List to store tiles within non-tree areas
    non_tree_areas_tiles = []

    # List to store tiles intersecting both tree and non-tree areas
    tiles_between_areas = []

    # List to store tiles within tree areas
    tree_area_tiles = []

    """**Select tiles likely to have trees**

    Selecting tiles likely to have trees is a vital step. To improve training efficiency and accuracy, it's best to choose tiles where trees are likely to be present. For instance, in the Porto Santo Project, out of 12,000 tiles, 9,000 were sea tiles, showing only the sea and no trees. To differentiate between tiles that can contain trees (‘land tiles’) and sea tiles, vector data from OpenStreetMap, representing the sea around Porto Santo, was used. In different projects, a similar approach might be needed for other types of tiles, such as desert tiles or tiles showing large lakes.

    In this step, a fixed script can be used alongside the user's action of creating polygons for areas without trees. Specifically, the user can utilize QGIS, a popular geographic information system software, to create polygons representing regions devoid of trees. By doing so, the software can then distinguish between areas with trees and those without, aiding in the accurate selection of tiles likely to have trees for further analysis and processing.
    """

    # Path to the shapefile containing polygons of non-tree areas (e.g., water bodies)
    shapefile_path_of_areas_non_tree_polygons = c['path']['shapefile_of_non_tree_areas']

    print ('tiles are devided to categories')
    # Load non-tree area polygons from the shapefile

    if (shapefile_path_of_areas_non_tree_polygons != ''):
        areas_non_tree_polygons = gpd.read_file(shapefile_path_of_areas_non_tree_polygons)
        # Iterate through each tile in the specified directory
        for tile_path in tqdm(glob.glob(os.path.join(TILES_OUTPUT_FOLDER , '*tif'))):
            # Check if the tile is within non-tree areas
            if is_tile_within_object(tile_path, areas_non_tree_polygons):
                non_tree_areas_tiles.append(tile_path)
            # Check if the tile intersects with both tree and non-tree areas
            elif is_tile_intersects_with_object(tile_path, areas_non_tree_polygons):
                tiles_between_areas.append(tile_path)
            # If the tile is outside non-tree areas, consider it within tree areas
            else:
                tree_area_tiles.append(tile_path)


    else:
        tree_area_tiles = glob.glob(os.path.join(TILES_OUTPUT_FOLDER,'*.tif'))
        #saving the paths' lists as a pickle files for future use.

    with open(os.path.join(CONST_DIR_INNER_DATA, 'tree_area_tiles.pkl'), 'wb') as f:
        pickle.dump(tree_area_tiles, f)
    
    with open(os.path.join(CONST_DIR_INNER_DATA, 'tiles_between_areas.pkl'), 'wb') as f:
        pickle.dump(tiles_between_areas, f)

    with open(os.path.join(CONST_DIR_INNER_DATA, 'non_tree_areas_tiles.pkl'), 'wb') as f:
        pickle.dump(non_tree_areas_tiles, f)

    # File path where you want to save the list
    txt_file_path = os.path.join(CONST_DIR_INNER_DATA, 'tree_area_tiles.txt')
    print (txt_file_path)
    # Open the file in write mode ('w')
    with open(txt_file_path, 'w') as f:
        # Iterate over the list and write each element to a new line
        for path in tree_area_tiles:
            f.write(str(path) + '\n')


    """**Model Training using Detectree Library (Pythonic Library):**

    For additional details about the Detectree library, you can visit the following link: https://github.com/martibosch/detectree. This repository provides in-depth information and resources related to the library, including documentation and usage guidelines.
    """

    with open(CONST_DIR_INNER_DATA + 'tree_area_tiles.pkl', 'rb') as f:
        tree_area_tiles = pickle.load(f)

    if (c['model']['train']):
        print ('split has been started')

        # Select the training tiles from the tiled aerial imagery dataset
        # Using the TrainingSelector class from the Detectree library
        ts = TrainingSelector(img_filepaths= tree_area_tiles)
        # Split the dataset into training and testing sets using the 'cluster-I' method
        split_df = ts.train_test_split(method='cluster-I')

        with open(os.path.join(CONST_DIR_INNER_DATA, 'split_df.pkl'), 'wb') as f:
            pickle.dump(split_df , f)
            
        """**Data Selection for labeling, Data lableing and Model Training:**

        The Detectree tree library automatically chooses 1% of the tiles that most accurately represent the entire set of tiles for user labeling. The 1% tiles should be labeled by the user.

        """

        # Directory to store the training tiles
        train_tiles_dir = c['path']['train_tiles_dir']

        # Create the directory if it doesn't exist
        os.makedirs(train_tiles_dir, exist_ok=True)

        # Copy training tiles to the specified directory
        for tile_path in split_df[split_df['train'] == True]['img_filepath']:
            shutil.copy(tile_path, train_tiles_dir + os.path.basename(tile_path))

        root_dir = ''
        images_dir = train_tiles_dir
        masks_dir = c['path']['masks_dir']
        os.makedirs(masks_dir, exist_ok= True)

        print ("""**Note!**

        Before continuing with the code you have to make masks of the training tiles (you can use qgis or gimp).

        The training tiles are waiting for you in the training tiles directory

        Create masks only for the tiles that has tree in them and save them in the masks directory **with the same name as the original tile**.

        The following code will automaticly detect missing masks and generate blank masks (for tiles without trees).
        """)