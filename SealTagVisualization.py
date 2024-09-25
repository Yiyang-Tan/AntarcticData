# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:15:15 2024

@author: Yiyang Tan
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath
import matplotlib.colors as mcolors
# from cartopy.io.shapereader import natural_earth
# import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
# from matplotlib.ticker import LogFormatter
import cmocean
# from cartopy.io.img_tiles import Stamen

# read .nc file
file_path = 'E:/1.PhD/Tasmania/DataVis_task/ct157-F118-20_all_prof.nc' 
ds = xr.open_dataset(file_path)
longitude = ds['LONGITUDE'].values
latitude = ds['LATITUDE'].values
julian_dates = ds['JULD'].values
julian_day = np.array([date.timetuple().tm_yday for date in julian_dates])

# tranfer julian to month
months = [date.month for date in julian_dates]
unique_months = sorted(set(months))

# mapping colorbar setting
cmap = plt.get_cmap('viridis', len(unique_months))
bounds = list(range(min(unique_months), max(unique_months) + 2))
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Map boundary
theta = np.linspace(0, 2*np.pi, 100) 
center, radius = [0.5, 0.5], 0.5 
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)  

# plotting in SouthPolarStereo
plt.figure(figsize=(12, 10), dpi=300)
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_boundary(circle, transform=ax.transAxes)
ax.set_extent([-180, 180, -90, -45], crs=ccrs.PlateCarree())
ax.coastlines()
ax.stock_img()
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor='none', facecolor='lightblue', alpha=0.5)
ax.gridlines(draw_labels=True)

# sampling plot
scatter = ax.scatter(longitude, latitude, c=months, cmap=cmap, norm=norm, s=10, transform=ccrs.PlateCarree())

# colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, ticks=unique_months)
cbar.set_label('Month')
cbar.ax.set_yticklabels([['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m-1] for m in unique_months])

plt.title('Distribution of Sampling Points by Month')
plt.tight_layout()


# plotting in small scale
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 80, -73, -48], crs=ccrs.PlateCarree())
ax.stock_img()
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor='none', facecolor='lightblue', alpha=0.5)
ax.gridlines(draw_labels=True)
scatter = ax.scatter(longitude, latitude, c=months, cmap=cmap, norm=norm, s=20, transform=ccrs.PlateCarree(), )
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, ticks=unique_months)
cbar.set_label('Month')
cbar.ax.set_yticklabels([['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m-1] for m in unique_months])
plt.title('Distribution of Sampling Points by Month')
plt.tight_layout()


# plot vertical distributio of CHL
chl_p = ds["CHLA_PROCESSED"].values
depth = ds["DEPTH_PROCESSED"].values
chl_p[chl_p<0]=np.nan
julian_day_grid = np.repeat(julian_day[:, np.newaxis], depth.shape[1], axis=1)

min_latitude_index = np.argmin(latitude)
min_latitude_julian = julian_day[min_latitude_index]
min_latitude_julian2 = julian_day[295]

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
mesh = ax1.pcolormesh(julian_day_grid, depth, chl_p, shading='auto', cmap=cmocean.cm.haline, 
                      norm=LogNorm(vmin=np.nanmin(chl_p) + 1e-4, vmax=np.nanmax(chl_p)))
# ax1.set_xlabel('Month', fontsize=14)
ax1.set_ylabel('Depth (m)', fontsize=14)
ax1.set_title('Chlorophyll A Concentration from Elephant Seal CTD', fontsize=16)
ax1.set_ylim(-250, 0)
ax1.tick_params(axis='y', labelsize=12)
# contour_levels = [0.1, 1, 10]
# contour = ax1.contour(julian_day_grid, depth, chl_p, levels=contour_levels, colors='black', linewidths=0.7)
# ax1.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
ax1.axvline(x=min_latitude_julian, color='red', linestyle='--', linewidth=1.5, label='Highest Latitude in summer')
ax1.axvline(x=min_latitude_julian2, color='red', linestyle='--', linewidth=1.5, label='Highest Latitude in autumn')
ax1.legend(fontsize=12, loc="lower right")
# ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
# ax1.set_xticks(np.linspace(julian_day_grid.min(), julian_day_grid.max(), 6))
# ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], fontsize=12)
ax2 = ax1.twiny()  # second y-axis
ax2.set_xlim(ax1.get_xlim())
num_labels = 9
lat_indices = np.linspace(0, len(latitude)-1, num_labels, dtype=int)
ax2.set_xticks(np.linspace(0, len(latitude)-1, num_labels))  # 使用纬度数据的索引作为刻度
ax2.set_xticklabels([f'{latitude[i]:.2f}' for i in lat_indices], fontsize=12)  # 设置刻度标签为纬度值
ax2.set_xlabel('Latitude', fontsize=14)

# latitude vs Julian Day
ax3.plot(julian_day, latitude, color='blue', marker='o', linestyle='-')
ax3.set_xlabel('Month', fontsize=14)
ax3.set_ylabel('Latitude', fontsize=14)
ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
ax3.set_xticks(np.linspace(julian_day_grid.min(), julian_day_grid.max(), 6))
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], fontsize=12)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.axvline(x=min_latitude_julian, color='red', linestyle='--', linewidth=1.5)
ax3.axvline(x=min_latitude_julian2, color='red', linestyle='--', linewidth=1.5)
# cbar = fig.colorbar(mesh, ax=[ax1, ax3], orientation='vertical', fraction=0.046, pad=0.04)
# cbar.set_label('Chlorophyll Concentration (Log10, $mg/m^3$)', fontsize=14)
# cbar.ax.tick_params(labelsize=12)
plt.tight_layout()


# plot vertical distribution of LIGHT
light = ds['LIGHT_PROCESSED'].values
light[light<0] = np.nan
valid_rows = ~np.isnan(light).all(axis=1)  # check NaN row
light_p_valid = light[valid_rows]  # extract valid rows
latitude_v = latitude[valid_rows]

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
mesh = ax1.pcolormesh(julian_day_grid[valid_rows], depth[valid_rows], light_p_valid, shading='auto', cmap=cmocean.cm.thermal)
# ax1.set_xlabel('Month', fontsize=14)
ax1.set_ylabel('Depth (m)', fontsize=14)
ax1.set_title('Photosynthetic photon flux density from Elephant Seal CTD', fontsize=16)
ax1.set_ylim(-250, 0)
ax1.tick_params(axis='y', labelsize=12)
# cbar = fig.colorbar(mesh, ax=ax1)
# cbar.set_label('Photosynthetic photon flux density $ln(µmol/m^{2}/s)$', fontsize=14)
# cbar.ax.tick_params(labelsize=12)
contour_levels = [5, 10]
contour = ax1.contour(julian_day_grid[valid_rows], depth[valid_rows], light_p_valid, levels=contour_levels, colors='black', linewidths=0.7)
ax1.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
ax1.axvline(x=min_latitude_julian, color='red', linestyle='--', linewidth=1.5)
ax1.axvline(x=min_latitude_julian2, color='red', linestyle='--', linewidth=1.5)
# ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
# ax1.set_xticks(np.linspace(julian_day_grid.min(), julian_day_grid.max(), 6))
# ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], fontsize=12)
ax2 = ax1.twiny()  
ax2.set_xlim(ax1.get_xlim()) 
num_labels = 8
lat_indices = np.linspace(0, len(latitude)-1, num_labels, dtype=int)
ax2.set_xticks(np.linspace(0, len(latitude)-1, num_labels))  
ax2.set_xticklabels([f'{latitude[i]:.2f}' for i in lat_indices], fontsize=12) 
ax2.set_xlabel('Latitude', fontsize=14)

# latitude vs Julian Day
ax3.plot(julian_day[valid_rows], latitude[valid_rows], color='blue', marker='o', linestyle='-')
ax3.set_xlabel('Month', fontsize=14)
ax3.set_ylabel('Latitude', fontsize=14)
ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
ax3.set_xticks(np.linspace(julian_day_grid.min(), julian_day_grid.max(), 6))
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], fontsize=12)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.axvline(x=min_latitude_julian, color='red', linestyle='--', linewidth=1.5, label='Highest Latitude in summer')
ax3.axvline(x=min_latitude_julian2, color='red', linestyle='--', linewidth=1.5, label='Highest Latitude in autumn')
ax3.legend(fontsize=12, loc="upper center")

plt.tight_layout()
