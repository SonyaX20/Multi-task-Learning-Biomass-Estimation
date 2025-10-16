"""
Data Preparation for Google Colab Training

Reads stacked TIF files, applies filtering, assigns labels from GeoJSON,
and compresses to HDF5 format.
"""

import os
import json
import h5py
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from collections import Counter


class DataPreparer:
    """Prepare training data with filtering"""
    
    def __init__(self, tif_dir, geojson_path, output_dir, chm_threshold=None):
        self.tif_dir = Path(tif_dir)
        self.geojson_path = Path(geojson_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # CHM threshold for filtering
        self.chm_threshold = chm_threshold
        
        # Load GeoJSON metadata
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Create filename to label and split mapping
        self.metadata = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for feature in geojson_data['features']:
            props = feature['properties']
            img_id = props['IMG_ID']
            bt_gen = props['BT_GEN']
            split = props['SPLIT']
            
            self.metadata[img_id] = {
                'label': bt_gen,
                'split': split
            }
        
        # Create label mapping
        unique_labels = sorted(set(meta['label'] for meta in self.metadata.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def process_tif_file(self, tif_path):
        """Process single TIF file with filtering"""
        try:
            with rasterio.open(tif_path) as src:
                data = src.read()  # Shape: (4, H, W)
            
            data = data.astype(np.float32)
            
            # Set -9999 to NaN for VV, VH, VV/VH_new (bands 1, 2, 4)
            for band_idx in [1, 2, 4]:
                mask = data[band_idx] == -9999.0
                if mask.any():
                    data[band_idx][mask] = np.nan
            
            # Apply CHM threshold filtering (band 0)
            if self.chm_threshold is not None:
                mask = data[0] > self.chm_threshold
                if mask.any():
                    data[0][mask] = np.nan
            
            return data
            
        except Exception as e:
            return None
    
    def get_img_id_from_filename(self, filename):
        """Extract IMG_ID from filename"""
        return filename.replace('.tif', '')
    
    def create_hdf5_dataset(self, compression='gzip', compression_level=4):
        """Create HDF5 datasets"""
        tif_files = sorted(list(self.tif_dir.glob('*.tif')))
        
        # Group by SPLIT field
        splits = {'train': [], 'test': []}
        
        for tif_path in tif_files:
            img_id = self.get_img_id_from_filename(tif_path.name)
            
            if img_id not in self.metadata:
                continue
            
            split = self.metadata[img_id]['split']
            if split in splits:
                splits[split].append(tif_path)
        
        # Create HDF5 file for each split
        for split_name, file_list in splits.items():
            if len(file_list) == 0:
                continue
            
            output_path = self.output_dir / f'{split_name}_data.h5'
            
            with h5py.File(output_path, 'w') as h5f:
                # Read first file to get dimensions
                first_data = self.process_tif_file(file_list[0])
                if first_data is None:
                    continue
                
                n_bands, height, width = first_data.shape
                n_samples = len(file_list)
                
                # Create datasets
                images_ds = h5f.create_dataset(
                    'images',
                    shape=(n_samples, n_bands, height, width),
                    dtype=np.float32,
                    compression=compression,
                    compression_opts=compression_level if compression == 'gzip' else None,
                    chunks=(1, n_bands, height, width)
                )
                
                labels_ds = h5f.create_dataset('labels', shape=(n_samples,), dtype=np.int32)
                
                # String arrays for filenames and label names
                dt = h5py.string_dtype(encoding='utf-8')
                filenames_ds = h5f.create_dataset('filenames', shape=(n_samples,), dtype=dt)
                label_names_ds = h5f.create_dataset('label_names', shape=(n_samples,), dtype=dt)
                
                # Process and write data
                valid_idx = 0
                for tif_path in tqdm(file_list, desc=f"Processing {split_name}"):
                    img_id = self.get_img_id_from_filename(tif_path.name)
                    
                    data = self.process_tif_file(tif_path)
                    if data is None:
                        continue
                    
                    label_name = self.metadata[img_id]['label']
                    label_idx = self.label_to_idx[label_name]
                    
                    images_ds[valid_idx] = data
                    labels_ds[valid_idx] = label_idx
                    filenames_ds[valid_idx] = tif_path.name
                    label_names_ds[valid_idx] = label_name
                    
                    valid_idx += 1
                
                # Resize if needed
                if valid_idx < n_samples:
                    images_ds.resize((valid_idx, n_bands, height, width))
                    labels_ds.resize((valid_idx,))
                    filenames_ds.resize((valid_idx,))
                    label_names_ds.resize((valid_idx,))
                
                # Save metadata
                h5f.attrs['n_samples'] = valid_idx
                h5f.attrs['n_bands'] = n_bands
                h5f.attrs['height'] = height
                h5f.attrs['width'] = width
                h5f.attrs['n_classes'] = len(self.label_to_idx)
                h5f.attrs['split'] = split_name
    
    def save_label_mapping(self):
        """Save label mapping"""
        mapping_path = self.output_dir / 'label_mapping.json'
        
        with open(mapping_path, 'w') as f:
            json.dump({
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label,
                'n_classes': len(self.label_to_idx)
            }, f, indent=2)
    
    def generate_statistics(self):
        """Generate dataset statistics"""
        stats = {
            'total_samples': len(self.metadata),
            'n_classes': len(self.label_to_idx),
            'class_distribution': {},
            'split_distribution': {}
        }
        
        for meta in self.metadata.values():
            label = meta['label']
            split = meta['split']
            
            if label not in stats['class_distribution']:
                stats['class_distribution'][label] = {'train': 0, 'test': 0, 'total': 0}
            
            stats['class_distribution'][label][split] = \
                stats['class_distribution'][label].get(split, 0) + 1
            stats['class_distribution'][label]['total'] += 1
            
            if split not in stats['split_distribution']:
                stats['split_distribution'][split] = 0
            stats['split_distribution'][split] += 1
        
        stats_path = self.output_dir / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    # Configuration paths
    tif_dir = Path(__file__).parent.parent / 'task_gen_chm' / 'chm_stacked_treesat_new'
    geojson_path = Path(__file__).parent / 'bb_60m.GeoJSON'
    output_dir = Path(__file__).parent / 'hdf5_data'
    
    # CHM threshold for filtering (adjust based on distribution_analysis.py results)
    chm_threshold = 45.23
    
    # Check if GeoJSON exists
    if not geojson_path.exists():
        original_geojson = Path(__file__).parent.parent / 'task_gen_chm' / 'bb_60m.GeoJSON'
        if original_geojson.exists():
            import shutil
            shutil.copy(original_geojson, geojson_path)
        else:
            print(f"Error: GeoJSON not found")
            return
    
    # Create data preparer
    preparer = DataPreparer(tif_dir, geojson_path, output_dir, chm_threshold)
    
    # Generate statistics
    preparer.generate_statistics()
    
    # Save label mapping
    preparer.save_label_mapping()
    
    # Create HDF5 datasets
    preparer.create_hdf5_dataset(compression='gzip', compression_level=4)
    
    print("Data preparation complete!")


if __name__ == '__main__':
    main()

