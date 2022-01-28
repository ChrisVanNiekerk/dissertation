#%%
# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Use geopandas for vector data and xarray for raster data
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import earthpy as et
import rasterio as rio
from rasterio.plot import show
# %%
# Define relative path to file
dtm_path = os.path.join("data","dtm",
                            "liDAR_1m_DTM_Clipped.tif")

dtm = rxr.open_rasterio(dtm_path)
# dtm
# # %%
# # View the Coordinate Reference System (CRS) & spatial extent
print("The CRS for this data is:", dtm.rio.crs)
print("The spatial extent is:", dtm.rio.bounds())
# # %%
# # View no data value
# print("The no data value is:", dtm.rio.nodata)
# # %%
# show(dtm)
# # %%
# dtm.plot.hist(color="purple")
# plt.show()
# # %%
# dtm.shape
# %%
# Open the data and mask no data values
# Squeeze reduces the third dimension given there is only one "band" or layer to this data
dtm_masked = rxr.open_rasterio(dtm_path, masked=True).squeeze()
# Notice there are now only 2 dimensions to your array
dtm_masked.shape
# %%
print("The DTM spatial extent is:", dtm_masked.rio.bounds())
print("The DTM CRS is:", dtm_masked.rio.crs)
# %%
# Plot the data and notice that the scale bar looks better
# No data values are now masked
f, ax = plt.subplots(figsize=(8, 8))
dtm_masked.plot(cmap="Greys_r",
                 ax=ax)
ax.set_title("DEM")
# ax.set_axis_off()
# show(dtm_masked, ax=ax)
plt.show()
# %%
# Open site boundary vector layer
site_bound_path = os.path.join("data","dtm",
                               "trial_buffer20m.shp")
site_bound_shp = gpd.read_file(site_bound_path)
# %%
site_bound_shp.head()
print("The site bound spatial extent is:", site_bound_shp.total_bounds)
print("The site bound CRS is:", site_bound_shp.crs)
# %%
site_bound_shp.geom_type
# %%
type(site_bound_shp)
# %%
site_bound_shp = site_bound_shp.to_crs(crs = 'EPSG:32735')
dtm_masked = dtm_masked.rio.reproject(dst_crs = 'EPSG:32735')
# site_bound_shp = site_bound_shp.to_crs('+proj=robin')
# %%
print("The DTM spatial extent is:", dtm_masked.rio.bounds())
print("The site bound spatial extent is:", site_bound_shp.total_bounds)
# %%
print(site_bound_shp.total_bounds)
# %%
# Plot the vector data
f, ax = plt.subplots(figsize=(8,8))
site_bound_shp.plot(color='teal',
                    edgecolor='black',
                    ax=ax)
ax.set(title="Site Boundary Layer - Shapefile")
plt.show()

# %%
f, ax = plt.subplots(figsize=(8, 8))


dtm_masked.plot.imshow(cmap="Greys", ax=ax, zorder=1)
site_bound_shp.plot(color='None',
                    edgecolor='teal',
                    linewidth=4,
                    ax=ax,
                    zorder=10)


ax.set(title="Raster Layer with Vector Overlay")
ax.axis('off')
plt.show()

# %%
dtm_masked.rio.resolution()

# %%
from shapely.geometry import mapping
lidar_clipped = dtm_masked.rio.clip(site_bound_shp.geometry.apply(mapping))

f, ax = plt.subplots(figsize=(10, 4))
lidar_clipped.plot(ax=ax)
ax.set(title="Raster Layer Cropped to Geodataframe Extent")
ax.set_axis_off()
plt.show()
# %%
pointcloud_raster_path = os.path.join("data","pointcloud",
                            "Riverdale Fl a1 Orth.tif")

pointcloud_raster = rxr.open_rasterio(pointcloud_raster_path)
pointcloud_raster
# %%
print("The DTM spatial extent is:", dtm_masked.rio.bounds())
print("The site bound spatial extent is:", site_bound_shp.total_bounds)
print("The pointcloud spatial extent is:", pointcloud_raster.rio.bounds(),"\n")
print("The pointcloud minimum raster value is: ", np.nanmin(pointcloud_raster.data))
print("The pointcloud maximum raster value is: ", np.nanmax(pointcloud_raster.data))
print("The pointcloud CRS of this data is:", pointcloud_raster.rio.crs)
print("The pointcloud resolution of this data is:", pointcloud_raster.rio.resolution())
# %%
pointcloud_raster.spatial_ref

# %%
dtm_masked_crs = dtm_masked.rio.crs
# pointcloud_raster = pointcloud_raster.rio.set_crs(4148)
# pointcloud_raster = pointcloud_raster.rio.set_crs(dtm_masked_crs, inplace=True)
pointcloud_raster = pointcloud_raster.rio.reproject(site_bound_shp.crs)
# %%



# %%
f, ax = plt.subplots(figsize=(8, 8))


dtm_masked.plot.imshow(cmap="Greys", ax=ax, zorder=1)
pointcloud_raster.plot.imshow(ax=ax, zorder=1)
site_bound_shp.plot(color='None',
                    edgecolor='teal',
                    linewidth=4,
                    ax=ax,
                    zorder=10)

ax.set(title="Raster Layer with Vector Overlay")
ax.axis('off')
plt.show()
a# %%
pointcloud_raster_path = os.path.join("data","pointcloud",
                            "Riverdale Fl a1 PTC.txt")
pc = pd.read_csv(pointcloud_raster_path)
# %%
pc_sample = pc.head(100)
# %%
pointcloud_sample_path = os.path.join("data","pointcloud",
                            "sample.txt")
pc_sample.to_csv(pointcloud_sample_path)
# %%

# %%
