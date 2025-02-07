import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from rasterio.windows import Window
from math import ceil
from operator import itemgetter

def split_and_visualize_tif(tif_path, output_dir='split_output', window_size=(2000, 2000)):
    """
    Split the entire TIF file into windows of specified size and visualize each part
    
    Args:
        tif_path: Path to the TIF file
        output_dir: Directory to save outputs
        window_size: (width, height) of each window in pixels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory to store data for all windows
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Store statistics for all windows
    windows_stats = []
    
    # Read the tree species codes and ensure ID is numeric
    species_codes = pd.read_csv('openagrar_derivate_00057802/tree_species_code.csv')
    species_codes['ID'] = pd.to_numeric(species_codes['ID'])
    
    with rasterio.open(tif_path) as src:
        # Get full image metadata
        height = int(src.height)  # Ensure it's an integer
        width = int(src.width)    # Ensure it's an integer
        print(f"Full image shape: ({height}, {width})")
        print(f"Coordinate system: {src.crs}")
        
        # Calculate number of windows needed
        n_rows = int(ceil(height / window_size[1]))
        n_cols = int(ceil(width / window_size[0]))
        print(f"\nSplitting image into {n_rows}x{n_cols} windows...")
        
        # Process each window
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate window coordinates
                x_start = int(col * window_size[0])
                y_start = int(row * window_size[1])
                
                # Adjust window size for image edges
                w_width = int(min(window_size[0], width - x_start))
                w_height = int(min(window_size[1], height - y_start))
                
                window = Window(x_start, y_start, w_width, w_height)
                
                try:
                    # Read the windowed data
                    raster_data = src.read(1, window=window)
                    
                    # Skip if window is empty or contains no valid data
                    if raster_data.size == 0 or not np.any(np.isin(raster_data, species_codes['ID'].values)):
                        continue
                    
                    print(f"\nProcessing window at coordinates ({x_start}, {y_start})")
                    
                    # Collect statistics for the window
                    window_info = {
                        'coordinates': (x_start, y_start),
                        'size': (w_width, w_height),
                        'species_distribution': {},
                        'invalid_percentage': 0,
                        'transform': src.window_transform(window),
                        'raster_data': raster_data
                    }
                    
                    # Calculate statistics
                    total_valid_pixels = 0
                    for _, species_row in species_codes.iterrows():
                        species_id = int(species_row['ID'])
                        species_count = np.sum(raster_data == species_id)
                        if species_count > 0:
                            total_valid_pixels += species_count
                            percentage = (species_count / raster_data.size) * 100
                            window_info['species_distribution'][species_row['species']] = {
                                'count': int(species_count),
                                'percentage': float(percentage)
                            }
                    
                    # Calculate invalid pixels
                    invalid_pixels = raster_data.size - total_valid_pixels
                    invalid_percentage = (invalid_pixels/raster_data.size)*100
                    window_info['invalid_percentage'] = float(invalid_percentage)
                    
                    # Save window info to temp directory
                    temp_path = os.path.join(temp_dir, f'window_{x_start}_{y_start}.pkl')
                    with open(temp_path, 'wb') as f:
                        pickle.dump(window_info, f)
                    
                    # Add to windows_stats list
                    windows_stats.append({
                        'coordinates': (x_start, y_start),
                        'invalid_percentage': invalid_percentage,
                        'temp_path': temp_path
                    })
                    
                except Exception as e:
                    print(f"Error processing window at ({x_start}, {y_start}): {str(e)}")
                    continue
        
        # Sort windows by invalid percentage and get top 15
        best_windows = sorted(windows_stats, key=itemgetter('invalid_percentage'))[:15]
        
        # Save final results for best 15 windows
        for idx, window_data in enumerate(best_windows):
            try:
                # Load window info from temp file
                with open(window_data['temp_path'], 'rb') as f:
                    window_info = pickle.load(f)
                
                x_start, y_start = window_info['coordinates']
                
                # Create visualization for best windows
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Plot raw data
                im1 = ax1.imshow(window_info['raster_data'])
                ax1.set_title(f'Raw Raster Data\nWindow: ({x_start}, {y_start})')
                plt.colorbar(im1, ax=ax1)
                
                # Create masked array for valid tree species
                valid_codes = species_codes['ID'].values.astype(np.int32)
                masked_data = np.ma.masked_array(
                    window_info['raster_data'],
                    ~np.isin(window_info['raster_data'], valid_codes)
                )
                
                # Plot masked data
                unique_species = np.sort(valid_codes)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_species)))
                cmap = plt.cm.colors.ListedColormap(colors)
                
                im2 = ax2.imshow(masked_data, cmap=cmap)
                ax2.set_title('Tree Species Distribution')
                cbar = plt.colorbar(im2, ax=ax2)
                
                # Add species labels
                tick_locs = valid_codes
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels(species_codes['species'])
                
                plt.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(output_dir, f'best_{idx+1}_viz_{x_start}_{y_start}.png')
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save statistics to text file
                stats_path = os.path.join(output_dir, f'best_{idx+1}_stats_{x_start}_{y_start}.txt')
                with open(stats_path, 'w') as f:
                    f.write(f"Window Statistics (Rank {idx+1}):\n")
                    f.write(f"Coordinates: ({x_start}, {y_start})\n")
                    f.write(f"Invalid pixel percentage: {window_info['invalid_percentage']:.2f}%\n\n")
                    f.write("Species Distribution:\n")
                    for species, data in window_info['species_distribution'].items():
                        f.write(f"{species}: {data['count']:,} pixels ({data['percentage']:.2f}%)\n")
                
            except Exception as e:
                print(f"Error processing best window at ({x_start}, {y_start}): {str(e)}")
                continue
        
        # Save all windows statistics
        stats_file = os.path.join(output_dir, 'all_windows_statistics.pkl')
        with open(stats_file, 'wb') as f:
            pickle.dump(windows_stats, f)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\nProcessing complete. Best 15 windows saved to {output_dir}")
        print(f"Full statistics saved to {stats_file}")

if __name__ == "__main__":
    try:
        tif_path = 'openagrar_derivate_00057802/Dominant_Species_Class.tif'
        split_and_visualize_tif(tif_path)
    except FileNotFoundError:
        print("Please provide the correct path to your TIF file")
    except rasterio.errors.RasterioIOError:
        print("Error reading the TIF file. Please check if the file is valid.") 