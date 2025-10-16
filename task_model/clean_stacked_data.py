"""
数据清洗脚本
根据分析结果对stacked_chm_sentinel数据进行清洗
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import json


class DataCleaner:
    """清洗堆叠的CHM和Sentinel数据"""
    
    def __init__(self, input_dir, output_dir, config=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 默认清洗配置
        self.config = config or {
            'chm': {
                'replace_inf_with_nan': True,
                'replace_negative_with_zero': True,
                'max_height_threshold': 80.0,  # 超过80m的树高视为异常
                'clip_values': True,
                'min_value': 0.0,
                'max_value': 80.0
            },
            'vv': {
                'replace_inf_with_nan': True,
                'clip_values': False,
                'min_value': -50.0,
                'max_value': 50.0
            },
            'vh': {
                'replace_inf_with_nan': True,
                'clip_values': False,
                'min_value': -50.0,
                'max_value': 50.0
            },
            'vv_vh_ratio': {
                'replace_inf_with_nan': True,
                'clip_values': True,
                'min_value': 0.0,
                'max_value': 100.0  # 过大的比值clip掉
            },
            'filter': {
                'max_nan_ratio': 0.5,  # 过滤NaN比例超过50%的样本
                'min_valid_pixels': 100  # 至少要有100个有效像素
            }
        }
        
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'filtered_files': 0,
            'cleaning_operations': {
                'inf_replaced': 0,
                'negatives_fixed': 0,
                'outliers_clipped': 0,
                'high_nan_filtered': 0
            }
        }
    
    def clean_chm_band(self, data):
        """清洗CHM波段"""
        cleaned = data.copy().astype(np.float32)
        config = self.config['chm']
        
        # 替换Inf为NaN
        if config['replace_inf_with_nan']:
            inf_mask = np.isinf(cleaned)
            if inf_mask.any():
                cleaned[inf_mask] = np.nan
                self.stats['cleaning_operations']['inf_replaced'] += inf_mask.sum()
        
        # 处理负值
        if config['replace_negative_with_zero']:
            negative_mask = cleaned < 0
            if negative_mask.any():
                cleaned[negative_mask] = 0.0
                self.stats['cleaning_operations']['negatives_fixed'] += negative_mask.sum()
        
        # Clip异常值
        if config['clip_values']:
            # 保留NaN
            valid_mask = ~np.isnan(cleaned)
            outlier_mask = valid_mask & ((cleaned < config['min_value']) | (cleaned > config['max_value']))
            if outlier_mask.any():
                cleaned = np.clip(cleaned, config['min_value'], config['max_value'])
                self.stats['cleaning_operations']['outliers_clipped'] += outlier_mask.sum()
        
        return cleaned
    
    def clean_sar_band(self, data, band_name):
        """清洗SAR波段（VV或VH）"""
        cleaned = data.copy().astype(np.float32)
        config = self.config[band_name.lower()]
        
        # 替换Inf为NaN
        if config['replace_inf_with_nan']:
            inf_mask = np.isinf(cleaned)
            if inf_mask.any():
                cleaned[inf_mask] = np.nan
                self.stats['cleaning_operations']['inf_replaced'] += inf_mask.sum()
        
        # 可选的clip
        if config.get('clip_values', False):
            valid_mask = ~np.isnan(cleaned)
            outlier_mask = valid_mask & ((cleaned < config['min_value']) | (cleaned > config['max_value']))
            if outlier_mask.any():
                cleaned = np.clip(cleaned, config['min_value'], config['max_value'])
                self.stats['cleaning_operations']['outliers_clipped'] += outlier_mask.sum()
        
        return cleaned
    
    def clean_ratio_band(self, data):
        """清洗VV/VH比值波段"""
        cleaned = data.copy().astype(np.float32)
        config = self.config['vv_vh_ratio']
        
        # 替换Inf为NaN
        if config['replace_inf_with_nan']:
            inf_mask = np.isinf(cleaned)
            if inf_mask.any():
                cleaned[inf_mask] = np.nan
                self.stats['cleaning_operations']['inf_replaced'] += inf_mask.sum()
        
        # Clip异常比值
        if config['clip_values']:
            valid_mask = ~np.isnan(cleaned)
            outlier_mask = valid_mask & ((cleaned < config['min_value']) | (cleaned > config['max_value']))
            if outlier_mask.any():
                cleaned = np.clip(cleaned, config['min_value'], config['max_value'])
                self.stats['cleaning_operations']['outliers_clipped'] += outlier_mask.sum()
        
        return cleaned
    
    def should_filter_file(self, data):
        """判断是否应该过滤掉该文件"""
        filter_config = self.config['filter']
        
        # 检查每个波段
        for band_idx in range(data.shape[0]):
            band_data = data[band_idx]
            total_pixels = band_data.size
            nan_count = np.isnan(band_data).sum()
            valid_count = total_pixels - nan_count
            nan_ratio = nan_count / total_pixels
            
            # NaN比例过高
            if nan_ratio > filter_config['max_nan_ratio']:
                return True, f"Band {band_idx+1} NaN ratio {nan_ratio:.2%} > threshold {filter_config['max_nan_ratio']:.2%}"
            
            # 有效像素太少
            if valid_count < filter_config['min_valid_pixels']:
                return True, f"Band {band_idx+1} valid pixels {valid_count} < threshold {filter_config['min_valid_pixels']}"
        
        return False, None
    
    def clean_file(self, input_path, output_path):
        """清洗单个文件"""
        try:
            # 读取数据
            with rasterio.open(input_path) as src:
                data = src.read()
                meta = src.meta.copy()
            
            # 检查波段数
            if data.shape[0] != 4:
                print(f"Warning: {input_path.name} has {data.shape[0]} bands, expected 4. Skipping.")
                return False
            
            # 清洗每个波段
            cleaned_data = np.zeros_like(data, dtype=np.float32)
            cleaned_data[0] = self.clean_chm_band(data[0])
            cleaned_data[1] = self.clean_sar_band(data[1], 'vv')
            cleaned_data[2] = self.clean_sar_band(data[2], 'vh')
            cleaned_data[3] = self.clean_ratio_band(data[3])
            
            # 检查是否需要过滤
            should_filter, reason = self.should_filter_file(cleaned_data)
            if should_filter:
                self.stats['filtered_files'] += 1
                self.stats['cleaning_operations']['high_nan_filtered'] += 1
                # print(f"Filtered: {input_path.name} - {reason}")
                return False
            
            # 更新元数据
            meta.update({
                'dtype': 'float32',
                'count': 4
            })
            
            # 写入清洗后的数据
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(cleaned_data)
            
            self.stats['processed_files'] += 1
            return True
            
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            return False
    
    def process_all_files(self):
        """处理所有文件"""
        tif_files = sorted(list(self.input_dir.glob('*.tif')))
        self.stats['total_files'] = len(tif_files)
        
        print(f"Found {len(tif_files)} files to process...")
        print(f"Output directory: {self.output_dir}")
        
        for file_path in tqdm(tif_files, desc="Cleaning files"):
            output_path = self.output_dir / file_path.name
            self.clean_file(file_path, output_path)
        
        # 打印统计信息
        self.print_summary()
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "="*80)
        print("数据清洗摘要")
        print("="*80)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功处理: {self.stats['processed_files']}")
        print(f"已过滤: {self.stats['filtered_files']}")
        print(f"保留比例: {self.stats['processed_files']/self.stats['total_files']*100:.2f}%")
        
        print("\n清洗操作统计:")
        print(f"  Inf值替换: {self.stats['cleaning_operations']['inf_replaced']:,} 像素")
        print(f"  负值修正: {self.stats['cleaning_operations']['negatives_fixed']:,} 像素")
        print(f"  异常值裁剪: {self.stats['cleaning_operations']['outliers_clipped']:,} 像素")
        print(f"  高NaN比例过滤: {self.stats['cleaning_operations']['high_nan_filtered']} 文件")
        
        print("\n清洗配置:")
        print(json.dumps(self.config, indent=2))
    
    def save_stats(self, output_path):
        """保存统计信息"""
        with open(output_path, 'w') as f:
            json.dump({
                'stats': self.stats,
                'config': self.config
            }, f, indent=2)
        print(f"\nStats saved to: {output_path}")


def main():
    # 配置路径
    input_dir = Path(__file__).parent.parent / 'task_gen_chm' / 'stacked_chm_sentinel'
    output_dir = Path(__file__).parent / 'cleaned_stacked_data'
    stats_output = Path(__file__).parent / 'analysis_results' / 'cleaning_stats.json'
    
    # 可以自定义清洗配置
    custom_config = {
        'chm': {
            'replace_inf_with_nan': True,
            'replace_negative_with_zero': True,
            'max_height_threshold': 80.0,
            'clip_values': True,
            'min_value': 0.0,
            'max_value': 80.0
        },
        'vv': {
            'replace_inf_with_nan': True,
            'clip_values': False  # 不clip SAR值，保持原始动态范围
        },
        'vh': {
            'replace_inf_with_nan': True,
            'clip_values': False
        },
        'vv_vh_ratio': {
            'replace_inf_with_nan': True,
            'clip_values': True,
            'min_value': 0.0,
            'max_value': 100.0
        },
        'filter': {
            'max_nan_ratio': 0.5,  # 过滤NaN比例超过50%的样本
            'min_valid_pixels': 100
        }
    }
    
    print("="*80)
    print("数据清洗程序")
    print("="*80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 创建清洗器
    cleaner = DataCleaner(input_dir, output_dir, config=custom_config)
    
    # 处理所有文件
    cleaner.process_all_files()
    
    # 保存统计
    stats_output.parent.mkdir(exist_ok=True, parents=True)
    cleaner.save_stats(stats_output)
    
    print("\n清洗完成！")


if __name__ == '__main__':
    main()

