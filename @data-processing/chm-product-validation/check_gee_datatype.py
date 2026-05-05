#!/usr/bin/env python3
"""
Check the data type of CHM products in Google Earth Engine.
"""

import ee

# Initialize Earth Engine
ee.Initialize()

# Load CHM datasets
meta_chm = ee.ImageCollection(
    "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
).mosaic()

eth_chm = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")

# Get band info
print("Meta CHM band info:")
meta_info = meta_chm.getInfo()
print(f"  Bands: {meta_info['bands']}")

print("\nETH CHM band info:")
eth_info = eth_chm.getInfo()
print(f"  Bands: {eth_info['bands']}")

# Sample a small region to check actual values
test_point = ee.Geometry.Point([13.0, 52.0])  # Germany
test_region = test_point.buffer(100)

print("\nSampling Meta CHM values:")
meta_sample = meta_chm.sample(test_region, scale=10, numPixels=10).getInfo()
if meta_sample['features']:
    for i, feat in enumerate(meta_sample['features'][:5]):
        print(f"  Sample {i+1}: {feat['properties']}")

print("\nSampling ETH CHM values:")
eth_sample = eth_chm.sample(test_region, scale=10, numPixels=10).getInfo()
if eth_sample['features']:
    for i, feat in enumerate(eth_sample['features'][:5]):
        print(f"  Sample {i+1}: {feat['properties']}")

print("\nChecking if toFloat() changes anything:")
eth_float = eth_chm.toFloat()
eth_float_sample = eth_float.sample(test_region, scale=10, numPixels=10).getInfo()
if eth_float_sample['features']:
    for i, feat in enumerate(eth_float_sample['features'][:5]):
        print(f"  Sample {i+1}: {feat['properties']}")
