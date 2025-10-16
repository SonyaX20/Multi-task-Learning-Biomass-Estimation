import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import rasterio
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

v = vars()
class GEDIdata:
    def __init__(self, type, folder_path):
        self.type = type
        self.label = self.type
        self.df = self._extract_feat(folder_path)
        self._gedi_is_unique()

        self.df_valid = None
        self._valid_mask()

        print(f"====== GEDI data: {self.label} ======")
        # 查看结果
        print("total data points:", self.df_valid.shape)
        print(f"min: {self.df_valid['feat'].min()}, max: {self.df_valid['feat'].max()}")
        print("example:")
        print(self.df_valid.head())

    def _extract_feat(self, folder_path):
        feat_list = []
        lat_list = []
        lon_list = []
        band_list = []

        beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 
                 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

        if self.type == "xvar":
            self.label = "Canopy Height"
            for filename in os.listdir(folder_path):
                if filename.endswith(".h5"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            for beam in beams:
                                if beam in f:
                                    feat = f[beam][self.type][:,1]
                                    lat = f[beam]['lat_lowestmode'][:]
                                    lon = f[beam]['lon_lowestmode'][:]
                                    if len(feat) == len(lat) == len(lon):
                                        feat_list.append(feat)
                                        lat_list.append(lat)
                                        lon_list.append(lon)
                                        band_list.append(np.full(len(feat), beam))
                                    else:
                                        print(f"incompatible sizes: {filename}, {beam}")
                    except Exception as e:
                        print(f"error with reading: {filename}, error: {e}")
        else:
            self.label = "agbd"
            for filename in os.listdir(folder_path):
                if filename.endswith(".h5"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            for beam in beams:
                                if beam in f:
                                    feat = f[beam][self.type][:]
                                    lat = f[beam]['lat_lowestmode'][:]
                                    lon = f[beam]['lon_lowestmode'][:]
                                    if len(feat) == len(lat) == len(lon):
                                        feat_list.append(feat)
                                        lat_list.append(lat)
                                        lon_list.append(lon)
                                        band_list.append(np.full(len(feat), beam))
                                    else:
                                        print(f"incompatible sizes: {filename}, {beam}")

                    except Exception as e:
                        print(f"error with reading: {filename}, error: {e}")

        feat_all = np.concatenate(feat_list)
        lat_all = np.concatenate(lat_list)
        lon_all = np.concatenate(lon_list)
        band_all = np.concatenate(band_list)

        combined_array = pd.DataFrame({
            'lat': lat_all,
            'lon': lon_all,
            'feat': feat_all.astype(float),
            'beam': band_all.astype(str)
        })
        return combined_array
    
    def _gedi_is_unique(self):
        lat_lon = self.df[['lat', 'lon']]
        _, unique_indices = np.unique(lat_lon, axis=0, return_index=True)
        unique_df = self.df.iloc[unique_indices]
        print(f"unique data points: {unique_df.shape}")
        return unique_df


    def _valid_mask(self):
        valid = (self.df['feat'] > -9999.0)
        self.df_valid = self.df[valid]
        # Add unique ID for each valid data point
        self.df_valid = self.df_valid.reset_index(drop=True)
        self.df_valid['id'] = self.df_valid.index
        print("valid data points:", self.df_valid.shape)
    
    
    def visualize_gedi_osm(self,buffer=0.2,point_size=0.1,zoom=10):
        lat = self.df_valid['lat']
        lon = self.df_valid['lon']
        feat = self.df_valid['feat']

        # Create figure and axis with OSM projection
        osm = cimgt.OSM()
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=osm.crs)

        # Set map extent with a small buffer
        ax.set_extent([lon.min()-buffer, lon.max()+buffer, 
                    lat.min()-buffer, lat.max()+buffer])

        # Add OSM tiles as background
        ax.add_image(osm, zoom, alpha=1)

        # Plot GEDI data points
        scatter = ax.scatter(lon, lat, c=feat, s=point_size, cmap='viridis', 
                            norm=plt.Normalize(vmin=feat.min(), vmax=feat.max()),
                            transform=ccrs.PlateCarree(), alpha=1)

        # Add colorbar
        cbar = plt.colorbar(scatter, label='AGBD')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False

        plt.title(f'GEDI {self.label} Values on OSM')
        plt.tight_layout()
        plt.show()



from scipy.stats import binned_statistic_2d
from folium.raster_layers import ImageOverlay
from folium import FeatureGroup, Polygon

def add_gedi_agbd_layer(gedi_agbd, m):
    mean_lat = gedi_agbd.df['lat'].mean()
    mean_lon = gedi_agbd.df['lon'].mean()
    # Get data from gedi_agbd
    lats = gedi_agbd.df['lat'].values
    lons = gedi_agbd.df['lon'].values
    agbd_values = gedi_agbd.df['feat'].values

    # Define the grid for 25m x 25m pixels
    # Convert lat/lon to approximate 25m grid
    # Note: This is an approximation, as 1 degree of latitude is ~111km
    lat_res = 25 / 111000  # 25m in degrees latitude
    lon_res = 25 / (111000 * np.cos(np.radians(mean_lat)))  # 25m in degrees longitude

    # Calculate grid boundaries
    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_min, lon_max = np.min(lons), np.max(lons)

    # Create grid edges
    lat_edges = np.arange(lat_min, lat_max + lat_res, lat_res)
    lon_edges = np.arange(lon_min, lon_max + lon_res, lon_res)

    # Bin the data into the grid using mean values
    result, _, _, _ = binned_statistic_2d(
        lats, lons, agbd_values, 
        statistic='mean', 
        bins=[lat_edges, lon_edges]
    )

    # Create a masked array for invalid values
    grid_data = np.ma.masked_invalid(result)

    # Get colormap - using YlGn (Yellow-Green) colormap which has bright yellow
    cmap = plt.colormaps.get_cmap('YlGn')
    min_val = np.nanmin(grid_data)
    max_val = np.nanmax(grid_data)

    # Convert the grid to an RGB image
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    rgba_img = cmap(norm(grid_data))

    # Create bounds for the image overlay [south, west, north, east]
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    # Create a feature group for the GEDI grid
    gedi_layer = FeatureGroup(name="GEDI AGBD Grid (25m x 25m)")

    # Add the image overlay to the map
    image_overlay = ImageOverlay(
        image=rgba_img,
        bounds=bounds,
        opacity=0.8,
        name="GEDI AGBD Grid"
    )
    image_overlay.add_to(gedi_layer)

    # Add the feature group to the map
    gedi_layer.add_to(m)
    return m

def add_polygon_layer(gdf, m):
    polygon_group = FeatureGroup(name="Tree Polygons")
    # Add polygons directly to the feature group
    for index, row in gdf.iterrows():
        coords = [
            [row['lat1'], row['lon1']],
            [row['lat2'], row['lon2']], 
            [row['lat3'], row['lon3']],
            [row['lat4'], row['lon4']],
            [row['lat1'], row['lon1']]  # Close the polygon
        ]
        Polygon(
            locations=coords,
            color='red',
            weight=1,
            fill=False
        ).add_to(polygon_group)

    # Add the feature group to the map
    polygon_group.add_to(m)
    return m

def add_s1_tree_layer(filtered_df, s1_files, m, size):
    # Plot each tif file
    count = 0
    
    s1_filenames = [f[:-4] for f in s1_files if f.endswith('.tif')]
    valid_ids = set(s1_filenames).intersection(set(filtered_df.IMG_ID.values))
    relevant_df = filtered_df[filtered_df.IMG_ID.isin(valid_ids)]
    
    # Batch process files instead of one by one
    for img_id, row in relevant_df.iterrows():
        if count >= size:
            break
            
        filename = f"{row['IMG_ID']}.tif"
        file_path = os.path.join('data/treesat/s1/60m', filename)
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            continue
        try:
            # Read and normalize the tif data
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read first band
                # Use faster normalization with numpy operations
                data_min, data_max = data.min(), data.max()
                if data_max > data_min:  # Avoid division by zero
                    data_normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                else:
                    data_normalized = np.zeros_like(data, dtype=np.uint8)
            
            # Create bounds from corner coordinates - compute once
            min_lat = min(row['lat1'], row['lat2'], row['lat3'], row['lat4'])
            min_lon = min(row['lon1'], row['lon2'], row['lon3'], row['lon4'])
            max_lat = max(row['lat1'], row['lat2'], row['lat3'], row['lat4'])
            max_lon = max(row['lon1'], row['lon2'], row['lon3'], row['lon4'])
            bounds = [[min_lat, min_lon], [max_lat, max_lon]]
            
            # Create polygon coordinates once
            coords = [
                [row['lat1'], row['lon1']],
                [row['lat2'], row['lon2']], 
                [row['lat3'], row['lon3']],
                [row['lat4'], row['lon4']],
                [row['lat1'], row['lon1']]  # Close the polygon
            ]
            
            # Use a static colormap instead of lambda function
            n_colors = 256
            colormap_array = np.zeros((n_colors, 4), dtype=np.float32)
            for i in range(n_colors):
                val = i/255.0
                colormap_array[i] = [val, val, val, 0.7]
            
            # Add image overlay
            ImageOverlay(
                data_normalized,
                bounds=bounds,
                colormap=lambda x: tuple(colormap_array[int(x)]),
                opacity=0.7,
                cross_origin=True,
                zindex=count+10  # Ensure proper layering
            ).add_to(m)

            # Add polygon outline
            Polygon(
                locations=coords,
                color='red',
                weight=1,
                fill=False
            ).add_to(m)
            
            count += 1
            
        except Exception as e:
            # Skip problematic files instead of breaking the entire process
            print(f"Error processing {filename}: {e}")
            continue
    
    return m 

