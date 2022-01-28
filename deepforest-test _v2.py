# %%

# %%
import os
from re import I
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import exposure
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import mapping
from scipy.spatial import distance
# from scipy.spatial.distance import cdist
from deepforest import main
import deepforest
from deepforest import preprocess
import slidingwindow
# from PIL import Image
from shapely.geometry import Point
import torch

import json
from os import listdir
from os.path import isfile, join
import zipfile
import datetime
from deepforest import get_data
from deepforest import visualize
from deepforest import utilities
import sys
import gc
# from pympler.tracker import SummaryTracker
# from pympler import muppy, summary

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
import xgboost as xgb
import time

# %%
def outer_trees(tree_pos_calc_df):

    # Identify outermost points
    south_side = pd.DataFrame(columns=['tree_id', 'tree_easting', 'tree_northing'])
    north_side = pd.DataFrame(columns=['tree_id', 'tree_easting', 'tree_northing'])
    west_side = pd.DataFrame(columns=['tree_id', 'tree_easting', 'tree_northing'])
    east_side = pd.DataFrame(columns=['tree_id', 'tree_easting', 'tree_northing'])
    for row in tree_pos_calc_df['Row'].unique():

        if row == tree_pos_calc_df['Row'].min():
            south_side = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]
            # south_side = pd.concat([outerpoints_df, all_trees_in_row[['tree_id', 'tree_easting','tree_northing']]])
        elif row == tree_pos_calc_df['Row'].max():
            north_side = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]
            north_side = north_side.iloc[::-1].reset_index(drop=True)
        else:
            all_trees_in_row = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]

            east_plot = all_trees_in_row['Plot'].min()
            west_plot = all_trees_in_row['Plot'].max()

            east_tree = all_trees_in_row[(all_trees_in_row['Plot'] == east_plot) & (all_trees_in_row['Tree no'] == 1)][['tree_id', 'tree_easting','tree_northing']]
            west_tree = all_trees_in_row[(all_trees_in_row['Plot'] == west_plot) & (all_trees_in_row['Tree no'] == 6)][['tree_id', 'tree_easting','tree_northing']]
            west_side = pd.concat([west_side, west_tree])
            east_side = pd.concat([east_side, east_tree])
            
    east_side = east_side.iloc[::-1].reset_index(drop=True)
    outerpoints_df = pd.concat([south_side, west_side, north_side, east_side]).reset_index(drop=True)
    outerpoints_df['geometry'] = outerpoints_df.apply(lambda x: Point((float(x.tree_easting), float(x.tree_northing))), axis=1)
    outerpoints_gp = gpd.GeoDataFrame(outerpoints_df, geometry='geometry')

    return outerpoints_gp, outerpoints_df

def tree_buffer_area(outer_trees_df, buffer_dist):

    # Create list of Easting and Northing points
    easting_point_list = list(outer_trees_df['geometry'].x) 
    northing_point_list = list(outer_trees_df['geometry'].y)

    # Create polygon from corner points and convert to gpd dataframe
    corner_trees_poly = Polygon(zip(easting_point_list, northing_point_list))
    corner_trees_poly_df = gpd.GeoDataFrame(index=[0], geometry=[corner_trees_poly])

    # Buffer polygon and convert to gpd dataframe
    corner_trees_poly_buffered = corner_trees_poly.buffer(buffer_dist) 

    buffered_poly_df = gpd.GeoDataFrame(index=[0], geometry=[corner_trees_poly_buffered])

    return corner_trees_poly_df, buffered_poly_df

def annotation_json_to_csv():

    all_annotations_df = pd.DataFrame()
    zip_files = [f for f in listdir('annotations') if isfile(join('annotations', f))]
    # print(zip_files)

    for zip_file in zip_files:

        zip_filename = zip_file
        extracted_folder_name = 'annotations/' + zip_filename.replace('.zip','')
        zip_path = 'annotations/' + zip_filename

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_name)
        
        if not [f for f in listdir(extracted_folder_name) if isfile(join(extracted_folder_name, f))]: continue
        
        with open(extracted_folder_name + '/annotations.json') as json_file:
            data = json.load(json_file)

        img_name_list = [f for f in listdir(extracted_folder_name + '/data') if isfile(join(extracted_folder_name + '/data', f))]

        for file in img_name_list:

            if '.png' in file: image_name = file

        image_path = 'df_crops/' + image_name

        annotations_df = pd.json_normalize(data[0]['shapes'])
        annotations_df = annotations_df[['points', 'label']]
        annotations_df[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(annotations_df.points.tolist())
        annotations_df['image_path'] = image_path
        annotations_df = annotations_df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']]

        all_annotations_df = pd.concat([all_annotations_df, annotations_df], ignore_index=True)

    annotations_csv_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_annotations.csv'
    annotations_csv_filepath = os.path.join("annotations","annotation csv files",annotations_csv_filename)
    # annotations_csv_filepath = 'annotations/annotation csv files/' + annotations_csv_filename

    all_annotations_df.to_csv(annotations_csv_filepath, index=False)

    return all_annotations_df, annotations_csv_filename, annotations_csv_filepath

def plot_predictions_from_df(df, img, colour = (255, 255, 0)):

    # Draw predictions on BGR 
    image = img[:,:,::-1]
    predicted_raster_image = visualize.plot_predictions(image, df, color=colour)

    return predicted_raster_image

def nearest_neighbor(tree, tree_locations_df):

    trees = np.asarray(tree_locations_df[['tree_easting', 'tree_northing']])
    # distances = cdist([current_tree], trees, 'euclidean').T
    tree = np.array(tree).reshape(-1, 1)
    distances = euclidean_distances(tree.T, trees)
    nn_idx = distances.argmin()
    distance_to_nn = distances.T[nn_idx][0]
    distance_to_nn_squared = (distances.T[nn_idx][0])**2
    tree_id = tree_locations_df.loc[nn_idx, 'tree_id']

    return distance_to_nn, distance_to_nn_squared, tree_id

def find_duplicates(tree_locations_pred_df):

    duplicates_series = tree_locations_pred_df.duplicated(subset=['tree_id_pred'], keep=False)
    duplicates_list_idx = list(duplicates_series[duplicates_series == True].index)
    duplicates_list = list(tree_locations_pred_df.loc[duplicates_list_idx, 'tree_id_pred'])
    duplicates_list = list(set(duplicates_list))

    ids_to_remove_pred_id = []
    for duplicate in duplicates_list:

        duplicate_idx = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'] == duplicate]['tree_id_pred_nn_dist'].idxmax()
        ids_to_remove_pred_id.append(duplicate_idx)

    tree_locations_pred_df.loc[ids_to_remove_pred_id,'tree_id_pred'] = np.nan

    return ids_to_remove_pred_id, tree_locations_pred_df

def actual_tree_data(sigma):

    tree_actual_df_path = 'data/EG0181T Riverdale A9b MAIN.xlsx'
    tree_actual_df = pd.read_excel(open(tree_actual_df_path, 'rb'), sheet_name='Data Rods versus Drone')
    last_valid_entry = tree_actual_df['Plot'].last_valid_index()
    tree_actual_df = tree_actual_df.loc[0:last_valid_entry]
    tree_actual_df = tree_actual_df.astype({'Plot':'int','Rep':'int','Tree no':'int'})
    tree_actual_df['tree_id'] = tree_actual_df['Plot'].astype('str') + '_' + tree_actual_df['Tree no'].astype('str')
    tree_actual_df = tree_actual_df[['tree_id', 'Plot', 'Rep', 'Tree no', 'Hgt22Rod','24_Def', 'Hgt22Drone']]
    tree_actual_df['Hgt22Rod'] = pd.to_numeric(tree_actual_df['Hgt22Rod'], errors='coerce').fillna(0)
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
    # min_height = tree_actual_df_no_dead['Hgt22Rod'].mean() - sigma * tree_actual_df_no_dead['Hgt22Rod'].std()
    min_height = tree_actual_df_no_dead['Hgt22Rod'].min()

    return tree_actual_df, tree_actual_df_no_dead, min_height

def pred_heights(df):

    # Sample the raster at every point location and store values in DataFrame
    pts = df[['X', 'Y', 'geometry']]
    pts.index = range(len(pts))
    coords = [(x,y) for x, y in zip(pts.X, pts.Y)]
    df['height_pred'] = [x[0] for x in src.sample(coords)]

    return df

def local_maxima_func(tree_actual_df, chm_clipped_path, window_size, min_height, save_shape_file=False):
    # Load CHM for local maxima
    with rasterio.open(chm_clipped_path) as source:
        chm_img = source.read(1) # Read raster band 1 as a numpy array
        affine = source.transform

    # Load CHM for height sampling
    src = rasterio.open(chm_clipped_path)

    coordinates = peak_local_max(chm_img, min_distance=window_size, threshold_abs=min_height)
    X=coordinates[:, 1]
    y=coordinates[:, 0]
    xs, ys = affine * (X, y)
    df_global = pd.DataFrame({'X':xs, 'Y':ys})
    df_local = pd.DataFrame({'X':X, 'Y':y})
    df_global['geometry'] = df_global.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
    df_global_gp = gpd.GeoDataFrame(df_global, geometry='geometry')

    if save_shape_file == True:
        file_name = 'lm_shape_files/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_lm_tree_points_' + str(window_size) + '.shp'
        df_global_gp.to_file(file_name, driver='ESRI Shapefile')

    # Sample CHM to obtain predicted heights
    df_global_gp = pred_heights(df_global_gp)

    # Filter trees < min height
    df_global_gp = df_global_gp[df_global_gp['height_pred'] >= min_height].reset_index(drop=True)
    tree_positions_from_lm = df_global_gp[['X','Y']]

    for idx in range(len(df_global_gp)):

        current_tree = (df_global_gp.loc[idx, 'X'], df_global_gp.loc[idx, 'Y'])
        distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
        df_global_gp.loc[idx, 'tree_id_pred'] = tree_id
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

    # Allocate predictions to actual trees
    ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(df_global_gp)

    # Merge with actual data to determine number of dead trees predicted
    tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
    
    results_df = pd.DataFrame()
    results_df.loc[results_idx, 'window_size'] = window_size
    results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
    results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
    results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
    results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
    results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
    results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
    results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
    results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()
    results_df.loc[results_idx, 'MAE_height'] = np.abs(tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred']).mean()
    results_df.loc[results_idx, 'MSE_height'] = np.square(np.abs(tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred'])).mean()
    results_df.loc[results_idx, 'RMSE_height'] = np.sqrt(results_df.loc[results_idx, 'MSE_height'])
    results_df.loc[results_idx, 'max_height_pred'] = tree_locations_pred_df['height_pred'].max()
    results_df.loc[results_idx, 'min_height_pred'] = tree_locations_pred_df['height_pred'].min()

    return df_global_gp, tree_positions_from_lm, results_df

def deep_forest_pred(tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False, classify_dead_trees = True):

    # Read in dataframe of all trees
    tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

    # Read in dataframe of all trees (shifted)
    tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
    tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
    tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
    tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

    results_df = pd.DataFrame()
    results_idx = 0
    # Create Predictions
    predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
    predictions_df = predictions_df[predictions_df['score'] > thresh]
    print(f"{predictions_df.shape[0]} predictions kept after applying threshold")

    predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)

    if save_fig == True: 
        df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
        plt.imsave(df_image_save_path,arr=predicted_raster_image)

    # Transform predictions to original CRS
    predictions_df_transform = predictions_df.copy()
    predictions_df_transform['image_path'] = "ortho_corrected_no_compression_clipped.tif"
    predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
    predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root, transform=True)

    predictions_df_transform['X'] = predictions_df_transform['xmin'] + (predictions_df_transform['xmax'] - predictions_df_transform['xmin'])/2
    predictions_df_transform['Y'] = predictions_df_transform['ymin'] + (predictions_df_transform['ymax'] - predictions_df_transform['ymin'])/2
    predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

    if save_shape == True:
        shape_file_name = 'deepforest_predictions/shapefiles/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
        predictions_df_transform.to_file(shape_file_name, driver='ESRI Shapefile')

    for idx in range(len(predictions_df_transform)):

        current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
        distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
        predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
        predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
        predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared
    
    # Add scores to transformed df
    predictions_df_transform['score'] = predictions_df['score']

    # Allocate predictions to actual trees
    ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

    # Merge with actual data to determine number of dead trees predicted
    tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

    # Save results
    results_df.loc[results_idx, 'patch_size'] = patch_size
    results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
    results_df.loc[results_idx, 'thresh'] = thresh
    results_df.loc[results_idx, 'iou_threshold'] = iou_threshold
    results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
    results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
    results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
    results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
    results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
    results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
    results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
    results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()

    results_idx += 1

    return predictions_df, predictions_df_transform, results_df, predicted_raster_image

def get_average_box_size(predictions_df_transform):

    # Ensure ymax and xmax are greater than ymin and xmin
    if predictions_df_transform['xmax'].mean() < predictions_df_transform['xmin'].mean():
        predictions_df_transform = predictions_df_transform.rename(columns={'xmin': 'xmax', 'xmax': 'xmin'})
    if predictions_df_transform['ymax'].mean() < predictions_df_transform['ymin'].mean():
        predictions_df_transform = predictions_df_transform.rename(columns={'ymin': 'ymax', 'ymax': 'ymin'})

    # Calculate average box size
    predictions_df_transform['width'] = predictions_df_transform['xmax'] - predictions_df_transform['xmin']
    predictions_df_transform['height'] = predictions_df_transform['ymax'] - predictions_df_transform['ymin']
    average_width = predictions_df_transform['width'].mean()
    average_height =  predictions_df_transform['height'].mean()
    average_side = (average_width + average_height) / 2

    return average_width, average_height, average_side

def boxes_from_points(tree_positions_for_classification_all, expansion_size):

    # Grow boxes from tree positions
    tree_positions_for_classification_all['xmin'] = tree_positions_for_classification_all['X'] - expansion_size / 2
    tree_positions_for_classification_all['ymin'] = tree_positions_for_classification_all['Y'] - expansion_size / 2
    tree_positions_for_classification_all['xmax'] = tree_positions_for_classification_all['X'] + expansion_size / 2
    tree_positions_for_classification_all['ymax'] = tree_positions_for_classification_all['Y'] + expansion_size / 2

    # Create polygons from expanded boxes
    geoms = []
    for idx in tree_positions_for_classification_all.index:

        xmin = tree_positions_for_classification_all.loc[idx,'xmin']
        xmax = tree_positions_for_classification_all.loc[idx,'xmax']
        ymin = tree_positions_for_classification_all.loc[idx,'ymin']
        ymax = tree_positions_for_classification_all.loc[idx,'ymax']    

        geom = Polygon([[xmin, ymin], [xmin,ymax], [xmax,ymax], [xmax,ymin]])
        geoms.append(geom)

    tree_positions_for_classification_all['geometry'] = geoms

    return tree_positions_for_classification_all

def dead_tree_classifier_dataset(predictions_df_transform, tree_actual_df, tree_positions_from_lm, chm_clipped_path, window_size=29, save_crops=False):

    # Filter DeepForest datatframe for tree points only
    tree_positions_from_df = predictions_df_transform[['X', 'Y']]

    # Isolate dead trees
    predictions_df_transform_incl_act = predictions_df_transform.merge(tree_actual_df, left_on='tree_id_pred', right_on='tree_id', how='left')
    predictions_df_transform_dead = predictions_df_transform_incl_act[predictions_df_transform_incl_act['24_Def'] == 'D'].reset_index(drop=True)
    tree_positions_from_df_dead_filtered = predictions_df_transform_dead[['X', 'Y']]
    # tree_positions_from_df_dead_filtered = tree_positions_from_df_dead_filtered.rename(columns={'X': 'X', 'Y': 'Y'})

    # Concatenate LocalMaxima and DeepForest outputs into one dataset
    tree_positions_all = pd.concat([tree_positions_from_lm, tree_positions_from_df]).reset_index(drop=True)

    # Randomly select trees from all trees (2 * number of dead trees)
    random_trees = np.random.randint(tree_positions_all.shape[0], size=tree_positions_from_df_dead_filtered.shape[0]*2)
    tree_positions_for_classification = tree_positions_all.loc[random_trees]

    # Create dataset
    tree_positions_for_classification_all = pd.concat([tree_positions_from_df_dead_filtered, tree_positions_for_classification]).reset_index(drop=True)

    # Expand points into boxes
    tree_positions_for_classification_all = boxes_from_points(tree_positions_for_classification_all, expansion_size)

    if save_crops == True:
        # Load ortho
        ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()

        crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_for_classification_all, save_crops = save_crops)
        crop_df.to_csv('tree_classifier_crops/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_rgb.csv')

        return tree_positions_for_classification_all, crop_df

    else: 
        return tree_positions_for_classification_all, None

def crop_array(ortho_cropped_for_cropping, tree_positions_for_classification_all, save_crops = False):
    
    # Create crops
    crop_df = pd.DataFrame()
    idx_counter = 0 
    for idx in tree_positions_for_classification_all.index:

        geo = tree_positions_for_classification_all.loc[idx,'geometry']
        clip = ortho_cropped_for_cropping.rio.clip([geo])
        clip = np.array(clip)
        clip = clip.astype('uint8')
        clip_rgb = np.moveaxis(clip, 0, 2).copy()
        tree_class_path = 'tree_classifier_crops/' + str(idx) + '.png'
        crop_df.loc[idx,'r_avg'] = clip_rgb[:,:,0].mean()
        crop_df.loc[idx,'g_avg'] = clip_rgb[:,:,1].mean()
        crop_df.loc[idx,'b_avg'] = clip_rgb[:,:,2].mean()

        if save_crops == True:
            plt.imsave(tree_class_path,arr=clip_rgb)

        if  (idx != 0) and (idx % 1000 == 0): 
            idx_counter += 1
            print(str(idx_counter*1000))

    return crop_df

def classification_scores(y_true, y_pred, y_pred_prob, model):

    score_df = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc'])

    score_df['confusion'] = score_df['confusion'].astype(object)
    score_df.loc[0,'model'] = model
    score_df.loc[0,'accuracy'] = accuracy_score(y_true, y_pred)
    score_df.loc[0,'f1'] = f1_score(y_true, y_pred)
    score_df.loc[0,'precision'] = precision_score(y_true, y_pred)
    score_df.loc[0,'recall'] = recall_score(y_true, y_pred)
    score_df.loc[0,'confusion'] = [confusion_matrix(y_true, y_pred)]
    score_df.loc[0,'auc'] = roc_auc_score(y_true, y_pred_prob) 

    return score_df
# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')
outerpoints_gp, outerpoints_df = outer_trees(tree_point_calc_shifted_df)
outer_trees_poly_df, outer_trees_buffered_poly_df = tree_buffer_area(outerpoints_gp, 1.5)
outer_trees_buffered_poly_df.to_file('ortho & pointcloud gen/outputs/GT/shape_files/boundary.shp', driver='ESRI Shapefile')

# Load, clip and save clipped ortho
ortho_raster_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "ortho_corrected_no_compression.tif")
# ortho = rxr.open_rasterio(ortho_raster_path, masked=True).squeeze()
ortho = rxr.open_rasterio(ortho_raster_path, masked=True)
ortho_clipped = ortho.rio.clip(outer_trees_buffered_poly_df.geometry)
ortho_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "ortho_corrected_no_compression_clipped.tif")
ortho_clipped.rio.to_raster(ortho_clipped_path)
# %%
ortho_clipped.shape
# %%
print("The CRS for this data is:", ortho_clipped.rio.crs)
print("The spatial extent is:", ortho_clipped.rio.bounds())
print("The no data value is:", ortho_clipped.rio.nodata)
# %%
ortho_clipped_path = "ortho & pointcloud gen/outputs/GT/ortho_corrected_no_compression_clipped.tif"
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
with rasterio.open(ortho_clipped_path) as source:
    img = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img = img.astype('float32')
img_rgb = np.moveaxis(img, 0, 2).copy()
img_bgr = img_rgb[...,::-1].copy()
# plt.imshow(img_rgb)

# %%
source
# %%
# windows = preprocess.compute_windows(img_rgb, 850, 0.5)
# for idx, window in enumerate(windows):

#     crop = window.apply(img_rgb)
#     deepforest.preprocess.save_crop('df_crops', 'ortho_cropped', idx, crop)
# %%
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv()
# %%
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1
# %%
# Set up model training
root_dir = ""
model.config["train"]["epochs"] = 100
model.config["workers"] = 0
# model.config["GPUS"] = 1
model.config["save-snapshot"] = False
model.config["train"]["csv_file"] = annotations_csv_filepath
model.config["train"]["root_dir"] = root_dir
# model.config["train"]['fast_dev_run'] = False
# %%
# Train model 
model.create_trainer()
model.trainer.fit(model)
model_path = 'df_models/100_epoch_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.pt'
torch.save(model.model.state_dict(),model_path)
# %%
model.model.load_state_dict(torch.load(model_path))
# %%
patch_size = 950
patch_overlap = 0.5
thresh = 0.8
iou_threshold = 0.05
# Create Predictions
predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
predictions_df = predictions_df[predictions_df['score'] > thresh]
print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)
plt.imsave(df_image_save_path,arr=predicted_raster_image)

# %%
# # Read in dataframe of all trees
# tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# # Import actual tree data from Sappi
# tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)

#  # Read in dataframe of all trees (shifted)
# tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'
# tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
# tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
# tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
# tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

# # patch_sizes = [700, 750, 800, 850, 900, 950]
# # patch_overlaps = [0.2, 0.3, 0.4, 0.5, 0.6]
# # thresholds = [0.2, 0.3, 0.4, 0.5]
# # iou_thresholds = [0.05, 0.1, 0.15, 0.2]

# patch_sizes = [950]
# patch_overlaps = [0.5]
# thresholds = [0.9]
# iou_thresholds = [0.05]

# results_idx = 0
# results_df = pd.DataFrame()
# for patch_size in patch_sizes:
#     for patch_overlap in patch_overlaps:
#         for thresh in thresholds:
#             for iou_threshold in iou_thresholds:

#                 try:
#                     # Create Predictions
#                     predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
#                     predictions_df = predictions_df[predictions_df['score'] > thresh]
#                     print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
#                     df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
#                     predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)
#                     plt.imsave(df_image_save_path,arr=predicted_raster_image)

#                     # Transform predictions to original CRS
#                     predictions_df_transform = predictions_df.copy()
#                     predictions_df_transform['image_path'] = "ortho_corrected_no_compression_clipped.tif"
#                     predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
#                     predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root, transform=True)

#                     predictions_df_transform['X'] = predictions_df_transform['xmin'] + (predictions_df_transform['xmax'] - predictions_df_transform['xmin'])/2
#                     predictions_df_transform['Y'] = predictions_df_transform['ymin'] + (predictions_df_transform['ymax'] - predictions_df_transform['ymin'])/2
#                     predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

#                     shape_file_name = 'deepforest_predictions/shapefiles/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
#                     predictions_df_transform.to_file(shape_file_name, driver='ESRI Shapefile')
                    
#                     clear_flag = 0
#                     for idx in range(len(predictions_df_transform)):

#                         if idx == range(len(predictions_df_transform))[-1]: clear_flag = 1

#                         current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
#                         distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df, clear_flag)
#                         predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
#                         predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
#                         predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

#                         del distance_to_nn
#                         del distance_to_nn_squared
#                         del tree_id

#                     # Allocate predictions to actual trees
#                     ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

#                     # Merge with actual data to determine number of dead trees predicted
#                     tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

#                     # Save results
#                     results_df.loc[results_idx, 'patch_size'] = patch_size
#                     results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
#                     results_df.loc[results_idx, 'thresh'] = thresh
#                     results_df.loc[results_idx, 'iou_threshold'] = iou_threshold
#                     results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
#                     results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
#                     results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
#                     results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
#                     results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
#                     results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
#                     results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
#                     results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()

#                     results_idx += 1
#                     print('test ' + str(results_idx) + ' done')

#                 except: 
#                     # Save results
#                     results_df.loc[results_idx, 'patch_size'] = patch_size
#                     results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
#                     results_df.loc[results_idx, 'thresh'] = thresh
#                     results_df.loc[results_idx, 'iou_threshold'] = iou_threshold


#                     gc.collect()

#                     results_idx += 1

# results_file_path = 'deepforest_predictions/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_predict_grid_search_results.csv'
# results_df.to_csv(results_file_path)

# %%
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'

patch_size = 950
patch_overlap = 0.5
thresh = 0.9
iou_threshold = 0.05

predictions_df, predictions_df_transform, results_df, predicted_raster_image = deep_forest_pred(tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False)
# %%
results_df
# %%
# Get tree positions from LocalMaxima
# Clipped CHM path
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "chm_2_clipped.tif")
df_global_gp, tree_positions_from_lm = local_maxima_func(chm_clipped_path, tree_actual_df, window_size=29, min_height=min_height, save_shape_file=True)
# %%
# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor
# %%
expansion_size
# %%
tree_positions_for_classification_all, crop_df = dead_tree_classifier_dataset(predictions_df_transform, tree_actual_df, tree_positions_from_lm, chm_clipped_path, window_size=29, save_crops=False)
# %%
classification_df = pd.read_csv('tree_classifier_crops/tree_crops_rgb_classified.csv')
classification_df
# %%
X = classification_df[['r_avg', 'g_avg', 'b_avg']]
Y = classification_df['class']
# %%
# Minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
# %%
scores_df_all_tests = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
for i in range(200):
    # Split dataset into test and train
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25, stratify=Y)
    # Instantiate model and fit
    from sklearn import svm

    rf = RandomForestClassifier(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
    ab = AdaBoostClassifier(n_estimators = 50, learning_rate = 1).fit(x_train, np.ravel(y_train))
    svm_model = svm.SVC(probability=True).fit(x_train, np.ravel(y_train))
    xgb_model = xgb.XGBClassifier(n_estimators = 100, objective="reg:squarederror", random_state=42, use_label_encoder=False).fit(x_train, np.ravel(y_train))

    # Make predictions
    predictions_rf = rf.predict(x_test)
    predictions_ab = ab.predict(x_test)
    predictions_xgb = xgb_model.predict(x_test)
    predictions_svm = svm_model.predict(x_test)
    predictions_rf_prob = rf.predict_proba(x_test)
    predictions_ab_prob = ab.predict_proba(x_test)
    predictions_svm_prob = svm_model.predict_proba(x_test)

    score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest')
    score_df_ab = classification_scores(y_true=y_test, y_pred=predictions_ab, y_pred_prob=predictions_ab_prob[:, 1], model='adaboost')
    score_df_xgb = classification_scores(y_true=y_test, y_pred=np.round(predictions_xgb), y_pred_prob=predictions_xgb, model='xgboost')
    score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm')

    scores_df_all_models = pd.concat([score_df_rf,score_df_ab,score_df_xgb,score_df_svm]).reset_index(drop=True)

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)
    
scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']] = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()

# %%
scores_df_all_tests_red.to_csv('tree_classifier_crops/classifier_scores/model_tests_all.csv')
scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_scores.csv')
# %%
scores_df_all_tests_avg
# %%
# Random Forest grid search
n_estimators = [10, 50, 100, 150, 200, 250, 300, 350]
criterions = ['gini', 'entropy']
max_depths = [None, 10, 20, 50, 100, 150, 200, 250]
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25, stratify=Y)

rf_parameter_search = pd.DataFrame(columns=['n_estimators', 'criteria', 'max_depth', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
grid_search_idx = 0
for n_estimator in n_estimators:
    for criterion in criterions:
        for max_depth in max_depths:

            rf = RandomForestClassifier(n_estimators = n_estimator, max_depth=max_depth, criterion = criterion, random_state=42).fit(x_train, np.ravel(y_train))
            predictions_rf = rf.predict(x_test)
            predictions_rf_prob = rf.predict_proba(x_test)
            score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest')

            rf_parameter_search.loc[grid_search_idx,'n_estimators'] = n_estimator
            rf_parameter_search.loc[grid_search_idx,'criteria'] = criterion
            rf_parameter_search.loc[grid_search_idx,'max_depth'] = max_depth
            rf_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc']] = score_df_rf.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc']]

            grid_search_idx += 1 
rf_parameter_search.to_csv('tree_classifier_crops/classifier_scores/random_forest_scores.csv')
# %%
# Build and save model (n_estimators = 50, criterion = 'gini, max_depth=None)
rf = RandomForestClassifier(n_estimators = 50, max_depth=None, criterion = 'gini', random_state=42).fit(x_train, np.ravel(y_train))
rf_model_filename = 'tree_classifier_crops/saved_models/random_forest.sav'
pickle.dump(rf, open(rf_model_filename, 'wb'))
# %%
################################################
#          TEST DEAD TREE CLASSIFIER           #
################################################
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# Clipped CHM path
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "chm_2_clipped.tif")

# Clipped ortho path
ortho_clipped_path = "ortho & pointcloud gen/outputs/GT/ortho_corrected_no_compression_clipped.tif"
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'

patch_size = 950
patch_overlap = 0.5
thresh = 0.9
iou_threshold = 0.05

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False)

# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor

# %%
# Get tree positions from LocalMaxima
df_global_gp, tree_positions_from_lm, results_df_lm = local_maxima_func(chm_clipped_path=chm_clipped_path, tree_actual_df=tree_actual_df, window_size=29, min_height=min_height, save_shape_file=True)
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']
print('total trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
# %%
df_global_gp_red = df_global_gp[['X', 'Y', 'tree_id_pred']]
df_global_gp_red['model'] = 'DF'
predictions_df_transform_red = predictions_df_transform[['X', 'Y', 'tree_id_pred']]
predictions_df_transform_red['model'] = 'LM'
all_trees_pred = pd.concat([df_global_gp_red, predictions_df_transform_red]).reset_index(drop=True)
# %%
tree_positions_expanded_for_classification_all = boxes_from_points(all_trees_pred, expansion_size)
# %%
# Load ortho
ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
# Crop all trees and obtain average R, G and B values for each crop
crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_expanded_for_classification_all, save_crops = False)
crop_df.to_csv('tree_classifier_crops/crops_file_from_all_trees/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_avg_rgb.csv')
# %%
# Scale features to match model input
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled_all_trees = scaler.transform(crop_df)
# %%
# Load tree classifier model
rf_model_filename = 'tree_classifier_crops/saved_models/random_forest.sav'
tree_classifier = pickle.load(open(rf_model_filename,"rb"))
# %%
# Make tree classification predictions
tree_classifications = rf.predict(X_scaled_all_trees)
tree_classifications_prob_dead = rf.predict_proba(X_scaled_all_trees)
all_trees_pred_incl_rgb = pd.concat([all_trees_pred,crop_df],axis=1)
all_trees_pred_incl_rgb['class'] = tree_classifications
all_trees_pred_incl_rgb['class_prob_dead'] = tree_classifications_prob_dead[:, 0]
# %%
all_trees_pred_incl_rgb[all_trees_pred_incl_rgb['class_prob_dead'] > 0.99]
# %%
pd.set_option('display.float_format', lambda x: '%.5f' % x)
all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['class'] == 0) & (all_trees_pred_incl_rgb['model'] == 'LM')]
# %%
df_global_gp[df_global_gp['tree_id_pred'] == '55_6']
# %%
all_trees_pred_incl_rgb[all_trees_pred_incl_rgb['tree_id_pred'] == '55_6']
# %%
all_trees_pred_incl_rgb.head()

# %%

# %%

# %%

# %%

# %%

# %%
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)
# %%
predictions_df_transform
# %%
# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor
# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
# %%

# %%
################################################
#               FINAL PIPELINE                 #
################################################

# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# Clipped CHM path
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "chm_2_clipped.tif")
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'

patch_size = 950
patch_overlap = 0.5
thresh = 0.9
iou_threshold = 0.05

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False)
# %%
# Get tree positions from LocalMaxima
df_global_gp, tree_positions_from_lm, results_df_lm = local_maxima_func(chm_clipped_path=chm_clipped_path, tree_actual_df=tree_actual_df, window_size=29, min_height=min_height)
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']
print('total trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
# %%
df_global_gp_red = df_global_gp[['X', 'Y', 'tree_id_pred']]
df_and_lm_trees = 
# %%
df_global_gp
# %%
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)
# %%
predictions_df_transform
# %%
# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor
# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
# %%
tree_positions_from_lm

# %%
predictions_df
# %%
rf_model_filename = 'tree_classifier_crops/saved_models/random_forest.sav'

 # Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

patch_size = 950
patch_overlap = 0.5
thresh = 0.5
iou_threshold = 0.05

results_df = pd.DataFrame()
results_idx = 0
# Create Predictions
predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
predictions_df = predictions_df[predictions_df['score'] > thresh]
print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)
plt.imsave(df_image_save_path,arr=predicted_raster_image)

# Transform predictions to original CRS
predictions_df_transform = predictions_df.copy()
predictions_df_transform['image_path'] = "ortho_corrected_no_compression_clipped.tif"
predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root, transform=True)

predictions_df_transform['X'] = predictions_df_transform['xmin'] + (predictions_df_transform['xmax'] - predictions_df_transform['xmin'])/2
predictions_df_transform['Y'] = predictions_df_transform['ymin'] + (predictions_df_transform['ymax'] - predictions_df_transform['ymin'])/2
predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

# shape_file_name = 'deepforest_predictions/shapefiles/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
# predictions_df_transform.to_file(shape_file_name, driver='ESRI Shapefile')

clear_flag = 0
for idx in range(len(predictions_df_transform)):

    if idx == range(len(predictions_df_transform))[-1]: clear_flag = 1

    current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df, clear_flag)
    predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

    del distance_to_nn
    del distance_to_nn_squared
    del tree_id

# Allocate predictions to actual trees
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

# Merge with actual data to determine number of dead trees predicted
tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

# Save results
results_df.loc[results_idx, 'patch_size'] = patch_size
results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
results_df.loc[results_idx, 'thresh'] = thresh
results_df.loc[results_idx, 'iou_threshold'] = iou_threshold
results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()

results_idx += 1
print('test ' + str(results_idx) + ' done')
# %%

# %%

# %%
test_clip = np.array(test_clip)
test_clip = test_clip.astype('uint8')
test_clip_rgb = np.moveaxis(test_clip, 0, 2).copy()
plt.imshow(test_clip_rgb)
plt.imsave('test.png',arr=test_clip_rgb)
# %%
test_clip_rgb
# %%
img_rgb
# %%
predictions_df_transform = predictions_df_transform.rename(columns={'geometry': 'tree_point'})
geoms = []
for idx in predictions_df_transform.index:
    xmin = predictions_df_transform.loc[idx,'xmin']
    xmax = predictions_df_transform.loc[idx,'xmax']
    ymin = predictions_df_transform.loc[idx,'ymin']
    ymax = predictions_df_transform.loc[idx,'ymax']    
    width = xmax - xmin
    height = ymax - ymin

    geom = Polygon([[xmin, ymin], [xmin,ymax], [xmax,ymax], [xmax,ymin]])
    geoms.append(geom)

predictions_df_transform['geometry'] = geoms
# %%
predictions_df_transform.head()
# %%

# %%

# %%

# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')
# outerpoints_gp, outerpoints_df = outer_trees(tree_point_calc_shifted_df)
# outer_trees_poly_df, outer_trees_buffered_poly_df = tree_buffer_area(outerpoints_gp, 0.8)
# outer_trees_buffered_poly_df.to_file('ortho & pointcloud gen/outputs/GT/shape_files/boundary.shp', driver='ESRI Shapefile')
# %%
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# %%
results_df = pd.DataFrame()
results_idx = 0
for idx in range(len(predictions_df_transform)):

    current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

# Allocate predictions to actual trees
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

# Merge with actual data to determine number of dead trees predicted
tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

# tree_locations_pred_df_cleaned = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()==False]
# results_df.loc[results_idx, 'window_size'] = size
results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()
# %%
results_df
# %%

# %%

# %%

# %%
predictions_df_transform
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
for patch_size in [850,900,950,1000]:

    predicted_raster = model.predict_tile(image=img_bgr, return_plot = True, patch_size=patch_size,patch_overlap=0.75)
    df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(patch_size) + '_from_bgr_75_overlap.png'
    plt.imsave(df_image_save_path,arr=predicted_raster)
    print(patch_size)
# %%

# %%
img = plt.figure(figsize = (20,20))
# plt.imshow(predicted_raster[:,:,::-1])
plt.imshow(img_bgr)
# plt.imshow(boxes[:,:,::-1])
# plt.show()
# %%
# Split raster into crops for training

# %%


# %%
model.config
# %%
os.path.dirname(annotations_csv_filepath)
# %%
_ROOT
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
plt.imsave("predicted_raster_overlap_0.1.png",arr=predicted_raster[:,:,::-1])
# predicted_raster[0][0]
# %%
annotations_file = get_data("annotations.csv")
# %%
model.config["epochs"] = 1
model.config["save-snapshot"] = False
model.config["train"]["csv_file"] = annotations_file
model.config["train"]["root_dir"] = os.path.dirname(annotations_file)
# model.config["train"]["root_dir"] = ""

model.create_trainer()

# %%
model.config["train"]["fast_dev_run"] = True
# %%
model.trainer.fit(model)
# %%
print(annotations_file)
# %%
