#!/usr/bin/env python3
"""
Analyze the distribution of train/test splits in bb_60m.GeoJSON 
and provide recommendations for validation set creation.
"""

import json
from collections import Counter, defaultdict
import numpy as np

def analyze_geojson_splits(geojson_file):
    """
    Analyze the train/test split distribution in the GeoJSON file.
    """
    print("Loading GeoJSON data...")
    with open(geojson_file, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    print(f"Total samples: {len(features)}")
    
    # Count train/test splits
    split_counts = Counter()
    species_by_split = defaultdict(lambda: defaultdict(int))
    akl_by_split = defaultdict(lambda: defaultdict(int))
    year_by_split = defaultdict(lambda: defaultdict(int))
    source_by_split = defaultdict(lambda: defaultdict(int))
    
    for feature in features:
        props = feature['properties']
        split = props['SPLIT']
        species = props['BT_BOT']  # Botanical name
        akl = props['AKL']  # Age class
        year = props['YEAR']
        source = props['SOURCE']
        
        split_counts[split] += 1
        species_by_split[split][species] += 1
        akl_by_split[split][akl] += 1
        year_by_split[split][year] += 1
        source_by_split[split][source] += 1
    
    # Basic statistics
    print("\n=== Basic Split Statistics ===")
    total = len(features)
    for split, count in split_counts.items():
        percentage = count / total * 100
        print(f"{split.capitalize()}: {count:,} samples ({percentage:.1f}%)")
    
    # Species distribution
    print("\n=== Species Distribution ===")
    all_species = set()
    for split_species in species_by_split.values():
        all_species.update(split_species.keys())
    
    print(f"Total unique species: {len(all_species)}")
    
    print("\nTop 10 species in each split:")
    for split in ['train', 'test']:
        print(f"\n{split.capitalize()} - Top 10 species:")
        sorted_species = sorted(species_by_split[split].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (species, count) in enumerate(sorted_species[:10], 1):
            print(f"  {i:2d}. {species}: {count:,}")
    
    # Age class distribution
    print("\n=== Age Class (AKL) Distribution ===")
    for split in ['train', 'test']:
        print(f"\n{split.capitalize()} AKL distribution:")
        sorted_akl = sorted(akl_by_split[split].items(), key=lambda x: x[0])
        for akl, count in sorted_akl:
            print(f"  AKL {akl}: {count:,}")
    
    # Year distribution
    print("\n=== Year Distribution ===")
    for split in ['train', 'test']:
        print(f"\n{split.capitalize()} year distribution:")
        sorted_years = sorted(year_by_split[split].items(), key=lambda x: x[0])
        for year, count in sorted_years:
            print(f"  {year}: {count:,}")
    
    # Source distribution
    print("\n=== Source Distribution ===")
    for split in ['train', 'test']:
        print(f"\n{split.capitalize()} source distribution:")
        sorted_sources = sorted(source_by_split[split].items(), 
                               key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources:
            print(f"  {source}: {count:,}")
    
    return {
        'total_samples': total,
        'split_counts': split_counts,
        'species_by_split': species_by_split,
        'akl_by_split': akl_by_split,
        'year_by_split': year_by_split,
        'source_by_split': source_by_split,
        'all_species': all_species
    }

def recommend_validation_strategy(analysis_result):
    """
    Provide recommendations for creating a validation set.
    """
    print("\n" + "="*60)
    print("VALIDATION SET CREATION RECOMMENDATIONS")
    print("="*60)
    
    total = analysis_result['total_samples']
    train_count = analysis_result['split_counts']['train']
    test_count = analysis_result['split_counts']['test']
    
    # Strategy 1: Split from training set
    print("\n### Strategy 1: Split from Training Set (Recommended)")
    print("Split the current training set into train/validation:")
    
    # Common validation ratios
    val_ratios = [0.1, 0.15, 0.2]
    
    print("\nProposed splits:")
    for val_ratio in val_ratios:
        new_train = int(train_count * (1 - val_ratio))
        new_val = train_count - new_train
        
        train_pct = new_train / total * 100
        val_pct = new_val / total * 100
        test_pct = test_count / total * 100
        
        print(f"  Validation ratio {val_ratio:.0%}:")
        print(f"    Train:      {new_train:,} ({train_pct:.1f}%)")
        print(f"    Validation: {new_val:,} ({val_pct:.1f}%)")
        print(f"    Test:       {test_count:,} ({test_pct:.1f}%)")
        print(f"    Ratio:      {train_pct:.0f}:{val_pct:.0f}:{test_pct:.0f}")
        print()
    
    # Strategy 2: Stratified sampling considerations
    print("\n### Strategy 2: Stratified Sampling Considerations")
    
    species_by_split = analysis_result['species_by_split']
    train_species = species_by_split['train']
    test_species = species_by_split['test']
    
    # Find species only in train or test
    train_only_species = set(train_species.keys()) - set(test_species.keys())
    test_only_species = set(test_species.keys()) - set(train_species.keys())
    common_species = set(train_species.keys()) & set(test_species.keys())
    
    print(f"Species distribution analysis:")
    print(f"  - Common species (train & test): {len(common_species)}")
    print(f"  - Train-only species: {len(train_only_species)}")
    print(f"  - Test-only species: {len(test_only_species)}")
    
    if train_only_species:
        print(f"\n  Train-only species (first 10): {list(train_only_species)[:10]}")
    if test_only_species:
        print(f"  Test-only species (first 10): {list(test_only_species)[:10]}")
    
    # Age class balance
    akl_by_split = analysis_result['akl_by_split']
    print(f"\n  Age class (AKL) considerations:")
    all_akl = set()
    for split_akl in akl_by_split.values():
        all_akl.update(split_akl.keys())
    
    for akl in sorted(all_akl):
        train_akl = akl_by_split['train'].get(akl, 0)
        test_akl = akl_by_split['test'].get(akl, 0)
        total_akl = train_akl + test_akl
        if total_akl > 0:
            train_ratio = train_akl / total_akl * 100
            test_ratio = test_akl / total_akl * 100
            print(f"    AKL {akl}: Train {train_ratio:.1f}% / Test {test_ratio:.1f}%")
    
    # Implementation recommendations
    print(f"\n### Strategy 3: Implementation Recommendations")
    print(f"""
    **Recommended Approach:**
    1. Use 15% validation split from training data (most balanced)
    2. Implement stratified sampling by:
       - Species (BT_BOT) - maintain species distribution
       - Age class (AKL) - maintain age representation
       - Source if relevant for your use case
    
    **Implementation steps:**
    1. Group training samples by species and age class
    2. Sample 15% from each group for validation
    3. Ensure minimum representation (e.g., ≥5 samples per class)
    4. For rare species/classes, consider oversampling or grouping
    
    **Final split suggestion (15% validation):**
    - Training:   ~{int(train_count * 0.85):,} samples (~{train_count * 0.85 / total * 100:.1f}%)
    - Validation: ~{int(train_count * 0.15):,} samples (~{train_count * 0.15 / total * 100:.1f}%)
    - Test:       {test_count:,} samples ({test_count / total * 100:.1f}%)
    
    **Code implementation:**
    ```python
    from sklearn.model_selection import train_test_split
    
    # Create stratification key combining species and age class
    strat_key = train_data['BT_BOT'] + '_' + train_data['AKL'].astype(str)
    
    # Split with stratification
    train_idx, val_idx = train_test_split(
        range(len(train_data)),
        test_size=0.15,
        stratify=strat_key,
        random_state=42
    )
    ```
    """)

def main():
    """Main analysis function."""
    geojson_file = "/Volumes/siyux1927/thesis/treesat_original/geojson/bb_60m.GeoJSON"
    
    # Analyze the current split
    analysis_result = analyze_geojson_splits(geojson_file)
    
    # Provide validation recommendations
    recommend_validation_strategy(analysis_result)

if __name__ == "__main__":
    main()
