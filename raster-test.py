#%%
import os
from re import I
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
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
from scipy.spatial.distance import cdist
#%%
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

def points_in_poly_filter(poly_df, points_df):

    filtered_points_df = sjoin(points_df, poly_df, how='left')
    filtered_points_df = filtered_points_df[filtered_points_df['index_right'].isna() != True]

    return filtered_points_df

def nearest_neighbor(tree, tree_locations_df):
    
    trees = np.asarray(tree_locations_df[['tree_easting', 'tree_northing']])
    distances = cdist([current_tree], trees).T
    nn_idx = distances.argmin()
    distance_to_nn = distances[nn_idx][0]
    distance_to_nn_squared = (distances[nn_idx][0])**2
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

    tree_actual_df_path = '../Data/EG0181T Riverdale A9b MAIN.xlsx'
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
# %%

# %%
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_no_dead, min_height = actual_tree_data(4)
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
outer_trees_poly_df, outer_trees_buffered_poly_df = tree_buffer_area(outerpoints_gp, 0.8)
outer_trees_buffered_poly_df.to_file('ortho & pointcloud gen/outputs/GT/shape_files/boundary.shp', driver='ESRI Shapefile')
# %%
# Import and clip raster
raster_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "chm_2.tif")
chm = rxr.open_rasterio(raster_path, masked=True).squeeze()
chm_clipped = chm.rio.clip(outer_trees_buffered_poly_df.geometry)
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "chm_2_clipped.tif")
chm_clipped.rio.to_raster(chm_clipped_path)
#%%
# Load CHM for local maxima
with rasterio.open(chm_clipped_path) as source:
    img = source.read(1) # Read raster band 1 as a numpy array
    affine = source.transform
img.shape

# Load CHM for height sampling
src = rasterio.open(raster_path)
# %%
results_df = pd.DataFrame()
results_idx = 0

for size in range(10,61):

    coordinates = peak_local_max(img, min_distance=size, threshold_abs=min_height)
    X=coordinates[:, 1]
    y=coordinates[:, 0]
    xs, ys = affine * (X, y)
    df_global = pd.DataFrame({'X':xs, 'Y':ys})
    df_local = pd.DataFrame({'X':X, 'Y':y})
    df_global['geometry'] = df_global.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
    df_global_gp = gpd.GeoDataFrame(df_global, geometry='geometry')

    # Sample CHM to obtain predicted heights
    df_global_gp = pred_heights(df_global_gp)

    # Filter trees < min height
    df_global_gp = df_global_gp[df_global_gp['height_pred'] >= min_height].reset_index(drop=True)
    # df_global_gp_filtered = points_in_poly_filter(outer_trees_buffered_poly_df, df_global_gp)

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

    # tree_locations_pred_df_cleaned = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()==False]
    results_df.loc[results_idx, 'window_size'] = size
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

    results_idx += 1
    print(size)

# %%
results_df.to_csv('window_size_test_results.csv', index=False)
# %%

tree_locations_pred_df_no_dead = tree_locations_pred_df[tree_locations_pred_df['24_Def'] != 'D']
np.sqrt(np.square(np.abs(tree_locations_pred_df_no_dead['Hgt22Rod'] - tree_locations_pred_df_no_dead['height_pred'])).mean())
# %%
tree_locations_pred_df
# %%
tree_locations_pred_df_cleaned['tree_id_pred_nn_dist'].mean()
# %%
# Read in calculated tree positions
# tree_locations_df_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_locations_df = pd.read_csv(tree_locations_df_path)
# distance_to_nn, tree_id = nearest_neighbor(tree, tree_locations_df)

# Read in predicted tree positions
tree_locations_pred_df_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_GT_2_30.shp'
tree_locations_pred_df = gpd.read_file(tree_locations_pred_df_path)
tree_locations_pred_df = tree_locations_pred_df[['X', 'Y', 'geometry']]

for idx in range(len(tree_locations_pred_df)):

    current_tree = (df_global_gp.loc[idx, 'X'], df_global_gp.loc[idx, 'Y'])
    distance_to_nn, tree_id = nearest_neighbor(current_tree, tree_locations_df)
    tree_locations_pred_df.loc[idx, 'tree_id_pred'] = tree_id
    tree_locations_pred_df.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
# %%

# %%
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(tree_locations_pred_df)
# %%

# %%
tuple(list(tree_rows[(tree_rows['id'] == ids) & (tree_rows['orientation'] == 'W')]['geometry'][69].coords)[0])
# %%
tree_rows[(tree_rows['id'] == ids) & (tree_rows['orientation'] == 'W')].index[0]
# %%
# tree_rows_compact.loc[idx,'geometry_w_point']
tree_rows_compact
# %%

# Read points from shapefile
tree_points_file_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_GT_230.shp'
pts = gpd.read_file(tree_points_file_path)
pts = pts[['X', 'Y', 'geometry']]
pts.index = range(len(pts))
coords = [(x,y) for x, y in zip(pts.X, pts.Y)]

# # Open the raster and store metadata
src = rasterio.open(raster_path)

# Sample the raster at every point location and store values in DataFrame
pts['Raster Value'] = [x for x in src.sample(coords)]
pts['Raster Value'] = probes.apply(lambda x: x['Raster Value'][0], axis=1)
# %%

pts_cropped = pts[(pts['X'] < 802845) & (pts['X'] > 802835)]
pts_cropped = pts_cropped[(pts_cropped['Y'] < 6695120) & (pts_cropped['Y'] > 6695110)]
# %%
pts_cropped
# %%
df_global
# %%

#%%
df_global
#%%
count = df_global['X'].count()
print('Total trees : {i}'.format(i = count))
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
cmap=plt.cm.gist_earth
ax = axes.ravel()
ax[0].imshow(img_rescale, cmap)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(img_rescale, cmap)
ax[2].autoscale(False)
ax[2].plot(X,y, 'r.')
ax[2].axis('off')
ax[2].set_title('Tree tops')

fig.tight_layout()
plt.show()
# %%
# ortho_path = os.path.join("ortho & pointcloud gen","outputs",
#                             "orthomosaic_flight_2.tif")
# im = plt.imread(ortho_path)
# implot = plt.imshow(im)

# plt.scatter(X,y)

# fig = plt.gcf()
# fig.set_size_inches(18.5*3, 10.5*3)

# plt.show()
# %%
# df_global_cropped = df_global[(df_global['X'] < 802845) & (df_global['X'] > 802835)]
# pts_cropped = pts[(pts['X'] < 802845) & (pts['X'] > 802835)]
# pts_cropped = pts_cropped[(pts_cropped['Y'] < 6695120) & (pts_cropped['Y'] > 6695110)]

# %%


# # Read in calculated tree positions
# tree_locations_df_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_locations_df = pd.read_csv(tree_locations_df_path)
# distance_to_nn, tree_id = nearest_neighbor(tree, tree_locations_df)

# # Read in predicted tree positions
# tree_locations_pred_df_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_GT_2_30.shp'
# tree_locations_pred_df = gpd.read_file(tree_locations_pred_df_path)
# tree_locations_pred_df = tree_locations_pred_df[['X', 'Y', 'geometry']]