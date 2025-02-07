# Tree Species Classification in German Forests

## How to use this .ipynb file

1️⃣ Download the dataset from the [OpenAgrar Repository](https://www.openagrar.de/receive/openagrar_mods_00084346)
2️⃣ Make sure requirements.txt is installed

## Dataset Overview

This project explores the dominant tree species classification dataset for Germany (2017/2018), which provides detailed spatial information about tree species distribution across German forests.

A tree species distribution map created using Sentinel-1 and Sentinel-2 satellite data combined with German National Forest Inventory (NFI) data:
- Sentinel-2 optical imagery (5-day intervals)
- Sentinel-1 SAR data (monthly composites)
- Environmental data (topography, meteorology, climate)
- German National Forest Inventory (NFI) data for validation

The mapping achieved high accuracy:
- 87.07 ± 0.3% for pure species stands
- 75.53 ± 0.07% for the entire map (including mixed stands)

Temporal coverage: 2017-2018 \
Geographic extent: Germany
- West boundary: 5.57°
- East boundary: 15.57°
- South boundary: 47.15°
- North boundary: 55.03°

## Dataset Structure

The dataset includes:
1. A TIFF file (`Dominant_Species_Class.tif`) containing the tree species classification
2. A CSV file (`tree_species_code.csv`) with species codes and names
3. Metadata in XML format

The classification includes 11 tree species groups:
- 9 specific tree species (Birch, Beech, Douglas fir, Oak, Alder, Spruce, Pine, Larch, Fir)
- 2 broader deciduous classes:
  - ODH: Other deciduous trees with high life expectancy
  - ODL: Other deciduous trees with low life expectancy

## Code Explanation (test.py)

The provided Python script (`test.py`) demonstrates how to work with the dataset:

### Key Functions:
1. `analyze_full_tif()`: 
   - Analyzes the complete TIF file
   - Provides distribution statistics of tree species
   - Identifies regions with valid data
   - Suggests optimal visualization windows

2. `visualize_and_crop_tif()`:
   - Creates visualizations of selected forest regions
   - Generates both raw and classified views
   - Saves cropped portions of the map
   - Calculates local species distribution statistics

### Visualization Results:
The code produces:
1. A PNG file showing:
   - Raw raster data
   - Classified tree species distribution
   - Color-coded species map with legend
2. A cropped TIF file of the selected region
3. Statistical analysis of species distribution

### Usage Notes:
- The code uses a windowing approach to handle the large dataset efficiently
- Custom colormaps are used to distinguish different species
- Invalid/No-data pixels (value 255) are masked out
- Species distribution statistics are calculated for each viewed region

## Source

Blickensdörfer, L., et al. (2024). "National tree species mapping using Sentinel-1/2 time series and German National Forest Inventory data." Remote Sensing of Environment, 304, 114069. DOI: 10.1016/j.rse.2024.114069

Dataset available at: [OpenAgrar Repository](https://www.openagrar.de/receive/openagrar_mods_00084346)