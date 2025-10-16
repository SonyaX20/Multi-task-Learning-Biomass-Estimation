#!/usr/bin/env python3
"""
Create visualizations for the forest dataset distribution analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_analyze_data(geojson_file):
    """Load and analyze the GeoJSON data."""
    print("Loading GeoJSON data...")
    with open(geojson_file, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    
    # Count train/test splits
    split_counts = Counter()
    species_by_split = defaultdict(lambda: defaultdict(int))
    
    for feature in features:
        props = feature['properties']
        split = props['SPLIT']
        species = props['BT_BOT']  # Botanical name
        
        split_counts[split] += 1
        species_by_split[split][species] += 1
    
    return split_counts, species_by_split

def create_train_test_pie_chart(split_counts):
    """Create pie chart for current train/test distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Training Set', 'Test Set']
    sizes = [split_counts['train'], split_counts['test']]
    colors = ['#3498db', '#e74c3c']
    
    # Create pie chart without explode and shadow
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, shadow=False,
                                     textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    # Add sample counts and percentages
    total = sum(sizes)
    for i, (label, size) in enumerate(zip(labels, sizes)):
        percentage = size / total * 100
        texts[i].set_text(f'{label}\n{size:,} samples\n({percentage:.1f}%)')
    
    ax.set_title('Current Dataset Distribution\nTrain vs Test Split', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)
    
    plt.tight_layout()
    return fig

def create_species_bar_chart(species_by_split):
    """Create grouped bar chart for species distribution."""
    # Get all species and their total counts
    all_species_counts = defaultdict(int)
    for split_data in species_by_split.values():
        for species, count in split_data.items():
            all_species_counts[species] += count
    
    # Sort species by total count (descending)
    sorted_species = sorted(all_species_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    species_names = [species for species, _ in sorted_species]
    train_counts = [species_by_split['train'].get(species, 0) for species, _ in sorted_species]
    test_counts = [species_by_split['test'].get(species, 0) for species, _ in sorted_species]
    total_counts = [train + test for train, test in zip(train_counts, test_counts)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(species_names))
    width = 0.7  # Wider bars for no gap
    
    # Create bars with no gap between them
    bars1 = ax.bar(x, train_counts, width, label='Training Set', color='#3498db', alpha=0.9)
    bars2 = ax.bar(x, test_counts, width, bottom=train_counts, label='Test Set', color='#e74c3c', alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Tree Species', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=16, fontweight='bold')
    ax.set_title('Species Distribution Across Training and Test Sets\n(Sorted by Total Sample Count)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation and remove padding
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in species_names], 
                       rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, len(species_names) - 0.5)  # Remove padding at start and end
    
    # Add legend with larger font
    legend = ax.legend(loc='upper right', fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Add value labels on bars with percentages
    def add_value_labels(bars, counts, total_counts):
        for bar, count, total in zip(bars, counts, total_counts):
            if count > 0:  # Only add label if bar has height
                height = bar.get_height()
                percentage = count / total * 100 if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2., height + max(total_counts)*0.05,
                       f'{int(count):,}\n({percentage:.1f}%)', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', rotation=0)
    
    add_value_labels(bars1, train_counts, total_counts)
    add_value_labels(bars2, test_counts, total_counts)
    
    # Set y-axis to start from 0 and add some padding
    ax.set_ylim(0, max(max(train_counts), max(test_counts)) * 1.25)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Increase tick label size
    ax.tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    return fig

def create_validation_split_pie_chart():
    """Create pie chart for recommended validation split."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Recommended split data
    labels = ['Training Set', 'Validation Set', 'Test Set']
    sizes = [37780, 7557, 5044]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Create pie chart without explode and shadow
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, shadow=False,
                                     textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    # Add sample counts to labels
    total = sum(sizes)
    for i, (label, size) in enumerate(zip(labels, sizes)):
        percentage = size / total * 100
        texts[i].set_text(f'{label}\n{size:,} samples\n({percentage:.1f}%)')
    
    ax.set_title('Recommended Dataset Distribution\n15% Validation Split with Stratified Sampling', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)
    
    # Add subtitle with ratios
    fig.text(0.5, 0.02, 'Final Ratio: 75% Train : 15% Validation : 10% Test', 
             ha='center', fontsize=14, style='italic', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_statistics_table(split_counts, species_by_split):
    """Create a summary table as text plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate statistics
    total_samples = sum(split_counts.values())
    train_samples = split_counts['train']
    test_samples = split_counts['test']
    
    # Recommended validation split
    val_samples = 7557
    new_train_samples = 37780
    
    # Species statistics
    total_species = len(set(list(species_by_split['train'].keys()) + 
                           list(species_by_split['test'].keys())))
    
    # Create summary text
    summary_text = f"""
DATASET ANALYSIS SUMMARY

Current Distribution:
• Total Samples: {total_samples:,}
• Training Set: {train_samples:,} samples ({train_samples/total_samples*100:.1f}%)
• Test Set: {test_samples:,} samples ({test_samples/total_samples*100:.1f}%)

Species Information:
• Total Unique Species: {total_species}
• All species present in both train and test sets
• Most common: Pinus sylvestris, Fagus sylvatica, Picea abies
• Includes "Cleared" category (harvested/disturbed areas)

Recommended Validation Split:
• Training Set: {new_train_samples:,} samples (75.0%)
• Validation Set: {val_samples:,} samples (15.0%)
• Test Set: {test_samples:,} samples (10.0%)
• Strategy: Stratified sampling by species and age class

Key Recommendations:
• Use stratified sampling to maintain species distribution
• Retain "Cleared" category for comprehensive land-use classification
• Implement minimum 5 samples per species-age combination
• Consider class weighting for imbalanced species
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_title('Forest Dataset Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    
    return fig

def main():
    """Main function to create all visualizations."""
    geojson_file = "/Volumes/siyux1927/thesis/treesat_original/geojson/bb_60m.GeoJSON"
    
    # Load and analyze data
    split_counts, species_by_split = load_and_analyze_data(geojson_file)
    
    # Create visualizations
    print("Creating train/test distribution pie chart...")
    fig1 = create_train_test_pie_chart(split_counts)
    fig1.savefig('current_train_test_distribution.png', dpi=300, bbox_inches='tight')
    
    print("Creating species distribution bar chart...")
    fig2 = create_species_bar_chart(species_by_split)
    fig2.savefig('species_distribution_by_split.png', dpi=300, bbox_inches='tight')
    
    print("Creating recommended validation split pie chart...")
    fig3 = create_validation_split_pie_chart()
    fig3.savefig('recommended_validation_split.png', dpi=300, bbox_inches='tight')
    
    print("Creating summary statistics...")
    fig4 = create_summary_statistics_table(split_counts, species_by_split)
    fig4.savefig('dataset_analysis_summary.png', dpi=300, bbox_inches='tight')
    
    # Show all plots
    plt.show()
    
    print("\nAll visualizations saved successfully!")
    print("Files created:")
    print("- current_train_test_distribution.png")
    print("- species_distribution_by_split.png")
    print("- recommended_validation_split.png")
    print("- dataset_analysis_summary.png")

if __name__ == "__main__":
    main()
