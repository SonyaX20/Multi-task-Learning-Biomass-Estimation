import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

def load_tif_file(filepath, band_idx):
    try:
        with rasterio.open(filepath) as src:
            data = src.read(band_idx + 1)
            data[data == -9999] = np.nan
            return data
    except:
        return None

def create_visualization(start_index=0):
    base_dir = Path(__file__).parent.parent
    stacked_dir = base_dir / 'chm_stacked_treesat_new'
    
    pattern = os.path.join(stacked_dir, "*.tif")
    all_files = sorted(glob.glob(pattern))
    sample_files = all_files[start_index:start_index+5]
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    band_indices = [0, 1, 2, 4]
    band_names = ['CHM', 'VV', 'VH', 'VV/VH']
    
    for col_idx, sample_file in enumerate(sample_files):
        filename = os.path.basename(sample_file).replace('.tif', '')
        
        for row_idx, band_idx in enumerate(band_indices):
            data = load_tif_file(sample_file, band_idx)
            
            if data is not None:
                im = axes[row_idx, col_idx].imshow(data, cmap='viridis', aspect='auto')
                cbar = plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
                cbar.set_label(band_names[row_idx], rotation=270, labelpad=15)
            
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(band_names[row_idx], fontsize=12, rotation=0, ha='right', va='center')
        
        axes[0, col_idx].set_title(filename, fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9)
    plt.savefig(f'stacked_bands_visualization_{start_index}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_visualization(1000)
