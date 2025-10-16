import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

def load_tif_file(filepath):
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            data[data == -9999] = np.nan
            return data
    except:
        return None

def create_visualization(start_index=0):
    base_dir = Path(__file__).parent.parent
    directories = {
        'dtm1_tif': base_dir / 'dtm1_tif',
        'dtm1_tif_smoothed': base_dir / 'dtm1_tif_smoothed', 
        'dsm1_tif': base_dir / 'dsm1_tif',
        'chm': base_dir / 'chm'
    }
    
    pattern = os.path.join(directories['dtm1_tif'], "*.tif")
    all_files = sorted(glob.glob(pattern))
    sample_files = all_files[start_index:start_index+5]
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    colormaps = ['gray', 'gray', 'gray', 'viridis']
    row_names = ['DTM1', 'DTM1 Smoothed', 'DSM1', 'CHM']
    
    for col_idx, sample_file in enumerate(sample_files):
        filename = os.path.basename(sample_file)
        
        for row_idx, (dir_name, dir_path) in enumerate(directories.items()):
            file_path = dir_path / filename
            data = load_tif_file(str(file_path))
            
            if data is not None:
                im = axes[row_idx, col_idx].imshow(data, cmap=colormaps[row_idx], aspect='auto')
                cbar = plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
                cbar.set_label(row_names[row_idx], rotation=270, labelpad=15)
            
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(row_names[row_idx], fontsize=12, rotation=0, ha='right', va='center')
        
        axes[0, col_idx].set_title(filename, fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9)
    plt.savefig(f'sample_visualization_{start_index}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_visualization(4980)
