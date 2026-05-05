import rasterio
import numpy as np

with rasterio.open('/Users/siyux1927/local/thesis0926/@data/chm-validation/eth_chm_10m/Acer_pseudoplatanus_1_142590_BI_NLF_eth_10m.tif') as src:
    data = src.read(1)
    print(f"Data type: {data.dtype}")  # 应该是 float32 或 float64
    print(f"Has decimals: {np.any(data % 1 != 0)}")  # 应该是 True
    print(f"Sample values: {data[data > 0][:5]}")  # 应该看到小数