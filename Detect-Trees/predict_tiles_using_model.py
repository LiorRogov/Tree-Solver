
import rasterio as rio
import numpy as np
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

with open('config.json', 'r') as f:
    c = json.load(f)

results = c['path']['tiles_results']


"""**Predict tree and grass locations and shapefile generation for each tile**

you should see the results in the results folder

"""

os.makedirs(c['path']['tiles_results'], exist_ok= True)

"""**Predict tree and grass locations (not necessary), sometimes it might be completely useless for example in the PortoSanto Project**

also, in order to use this part of the algorithm you have to make sure that the items in the following shapefile ('multipolygons') are MultiPolygon type class.
you can learn more here: https://geopandas.org/en/stable/docs/user_guide/data_structures.html
"""

"""**Create one shapefile of tree locations**"""

def clean_false_points(current_tile_with_trees ,shapefile_where_trees_cannot_exist):
    # Initialize a boolean mask for false points, initially all True
    true_points_mask = np.ones(len(current_tile_with_trees), dtype=bool)

    # Iterate through rows in the shapefile_where_trees_cannot_exist DataFrame
    for row in shapefile_where_trees_cannot_exist.itertuples():
        # Extract the polygon geometry from the row
        polygon = row[-1]

        # Iterate through rows in the current_tile_with_trees DataFrame
        for tile_points_row in current_tile_with_trees.itertuples():
            # Extract the point geometry and its row index from the row
            point = tile_points_row[-1]
            point_index = tile_points_row[0]

            # Check if the point is within the polygon
            if point.within(polygon):
                # If it is within the polygon, append its row index to the list
                true_points_mask[point_index] = False

    # Return a new DataFrame containing only rows not in the false_points_index
    return current_tile_with_trees.iloc[true_points_mask]

if (c['path']['shapefile_of_tiny_water_bodies_or_smalll_area_without_trees'] != ''):
    # Read the shapefile containing polygons where trees cannot exist
    shapefile_of_water_bodies_or_locations_where_trees_cannot_exist = gpd.read_file(c['path']['shapefile_of_tiny_water_bodies_or_smalll_area_without_trees'])


os.makedirs(c['path']['output'], exist_ok = True)
# Create an empty GeoDataFrame to store the resulting tree points
shapefile_trees = gpd.GeoDataFrame()

# Iterate through folders and files in the 'results' directory
for folder, _, data in tqdm(os.walk(f'{results}')):
    if len(data) != 0:
        # Read the GeoDataFrame containing all tree points for the current tile
        all_tile_points = gpd.read_file(f'{glob.glob(folder + "/*trees.shp")[0]}')
        
        if (c['path']['shapefile_of_tiny_water_bodies_or_smalll_area_without_trees'] != ''):
            # Clean false points from the current tile using the provided function
            true_tile_point = clean_false_points(all_tile_points, shapefile_of_water_bodies_or_locations_where_trees_cannot_exist)

            # Concatenate the resulting true tree points to the shapefile_trees
            shapefile_trees = pd.concat([shapefile_trees, true_tile_point])
        else:
            shapefile_trees = pd.concat([shapefile_trees, all_tile_points])

# Save the final shapefile containing the cleaned tree points
shapefile_trees.to_file(c['path']['output'] + r'trees.shp')

"""**Create one shapefile of grass locations**

"""

# Initialize an empty GeoDataFrame to store the grass polygons
shapefile_grass = gpd.GeoDataFrame()

# Iterate through folders and files in the 'results' directory
for folder, _, data in tqdm(os.walk(f'{results}')):
    if len(data) != 0:
        try:
            # Concatenate the GeoDataFrame from the current folder's 'grass.shp' file
            shapefile_grass = pd.concat([shapefile_grass, gpd.read_file(f'{folder}\\grass.shp')])
        except:
            print (folder)

# Save the final shapefile containing the cleaned grass polygons

shapefile_grass.to_file(c['path']['output'] + r'grass.shp')