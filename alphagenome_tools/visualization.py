"""
Visualization utilities for AlphaGenome results

This module provides convenient functions for visualizing AlphaGenome predictions,
including variant comparisons, batch summaries, and expression heatmaps.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")

# AlphaGenome imports
try:
    from alphagenome.visualization import plot_components
    from alphagenome.data import genome
except ImportError:
    plot_components = None
    genome = None

# Pandas
try:
    import pandas as pd
except ImportError:
    pd = None

# Setup logging
logger = logging.getLogger(__name__)

# Set default style
if MATPLOTLIB_AVAILABLE:
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10


def quick_plot(outputs, figsize=(12, 6), save_path: Optional[Path] = None):
    """
    Generate a quick preview plot of AlphaGenome outputs.

    This is a simplified visualization for quick inspection of results.

    Args:
        outputs: AlphaGenome prediction outputs
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Extract RNA-seq data if available
    data_plotted = False
    
    if hasattr(outputs, 'rna_seq') and outputs.rna_seq is not None:
        rna_data = outputs.rna_seq
        if hasattr(rna_data, 'values') and rna_data.values is not None:
            # Average across all tracks (cell types/tissues)
            mean_expression = np.mean(rna_data.values, axis=1)
            
            # Create x-axis (genomic positions)
            interval = rna_data.interval
            positions = np.linspace(interval.start, interval.end, len(mean_expression))
            
            # Plot mean expression
            ax.plot(positions, mean_expression, linewidth=1, alpha=0.8, color='steelblue')
            ax.fill_between(positions, mean_expression, alpha=0.3, color='steelblue')
            
            ax.set_xlabel(f'Genomic Position ({interval.chromosome})')
            ax.set_ylabel('Mean RNA Expression')
            ax.set_title(f'AlphaGenome RNA-seq Prediction\n{interval.chromosome}:{interval.start:,}-{interval.end:,}')
            ax.grid(True, alpha=0.3)
            data_plotted = True
    
    # If no data was plotted, show placeholder
    if not data_plotted:
        ax.text(0.5, 0.5, 'No RNA-seq data available in outputs',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('Genomic Position')
        ax.set_ylabel('Prediction Score')
        ax.set_title('AlphaGenome Prediction Results')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_variant_comparison(
    ref_outputs,
    alt_outputs,
    variant: genome.Variant,
    figsize=(14, 6),
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Create side-by-side comparison of reference and alternate allele predictions.

    Args:
        ref_outputs: Reference allele outputs
        alt_outputs: Alternate allele outputs
        variant: The variant being compared
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract RNA-seq data for both alleles
    ref_plotted = False
    alt_plotted = False
    
    # Plot reference allele
    if hasattr(ref_outputs, 'rna_seq') and ref_outputs.rna_seq is not None:
        rna_data = ref_outputs.rna_seq
        if hasattr(rna_data, 'values') and rna_data.values is not None:
            mean_expression = np.mean(rna_data.values, axis=1)
            interval = rna_data.interval
            positions = np.linspace(interval.start, interval.end, len(mean_expression))
            
            ax1.plot(positions, mean_expression, linewidth=1.5, alpha=0.8, color='steelblue')
            ax1.fill_between(positions, mean_expression, alpha=0.3, color='steelblue')
            
            # Mark variant position
            ax1.axvline(variant.position, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Variant')
            
            ax1.set_xlabel(f'Genomic Position ({interval.chromosome})', fontsize=10)
            ax1.set_ylabel('Mean RNA Expression', fontsize=10)
            ax1.set_title(f'Reference Allele ({variant.reference_bases})', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ref_plotted = True
    
    # Plot alternate allele
    if hasattr(alt_outputs, 'rna_seq') and alt_outputs.rna_seq is not None:
        rna_data = alt_outputs.rna_seq
        if hasattr(rna_data, 'values') and rna_data.values is not None:
            mean_expression = np.mean(rna_data.values, axis=1)
            interval = rna_data.interval
            positions = np.linspace(interval.start, interval.end, len(mean_expression))
            
            ax2.plot(positions, mean_expression, linewidth=1.5, alpha=0.8, color='coral')
            ax2.fill_between(positions, mean_expression, alpha=0.3, color='coral')
            
            # Mark variant position
            ax2.axvline(variant.position, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Variant')
            
            ax2.set_xlabel(f'Genomic Position ({interval.chromosome})', fontsize=10)
            ax2.set_ylabel('Mean RNA Expression', fontsize=10)
            ax2.set_title(f'Alternate Allele ({variant.alternate_bases})', fontsize=12, fontweight='bold', color='darkred')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            alt_plotted = True
    
    # If no data plotted, show placeholder
    if not ref_plotted:
        ax1.text(0.5, 0.5, f'Reference\n{variant.reference_bases}\nNo data',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Reference Allele')
        ax1.set_xlabel('Genomic Position')
        ax1.set_ylabel('Prediction Score')
    
    if not alt_plotted:
        ax2.text(0.5, 0.5, f'Alternate\n{variant.alternate_bases}\nNo data',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='red')
        ax2.set_title('Alternate Allele')
        ax2.set_xlabel('Genomic Position')
    ax2.set_ylabel('Prediction Score')

    # Set overall title
    if title is None:
        title = f'Variant Effect: {variant.chromosome}:{variant.position} ' \
                f'{variant.reference_bases}>{variant.alternate_bases}'

    fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_batch_summary(
    results_df: pd.DataFrame,
    metric: str = 'success',
    figsize=(10, 6),
    save_path: Optional[Path] = None
):
    """
    Create summary visualization for batch predictions.

    Args:
        results_df: DataFrame with batch prediction results
        metric: Column name to visualize
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    if pd is None:
        logger.error("pandas not available")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Count successes vs failures
    if metric == 'success':
        success_counts = results_df['success'].value_counts()
        colors = ['green' if x else 'red' for x in success_counts.index]
        success_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Batch Prediction Summary')
        
        # Set x-tick labels based on actual data
        labels = ['Success' if idx else 'Failure' for idx in success_counts.index]
        ax.set_xticklabels(labels, rotation=0)

        # Add count labels on bars
        for i, (idx, count) in enumerate(success_counts.items()):
            label = 'Success' if idx else 'Failure'
            ax.text(i, count, str(count), ha='center', va='bottom')

    # Chromosome distribution
    elif 'chromosome' in results_df.columns:
        chr_counts = results_df['chromosome'].value_counts().sort_index()
        chr_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('Count')
        ax.set_title('Variants by Chromosome')
        ax.set_xlabel('Chromosome')
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_expression_heatmap(
    data,
    figsize=(12, 8),
    cmap: str = 'viridis',
    save_path: Optional[Path] = None,
    title: str = 'Gene Expression Heatmap'
):
    """
    Create a heatmap visualization of gene expression predictions.

    Args:
        data: Expression data (2D array or DataFrame)
        figsize: Figure size (width, height)
        cmap: Colormap name
        save_path: Optional path to save the figure
        title: Figure title

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data = data.values
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expression Level')

    # Set labels
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Samples/Genes')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_tracks_overlaid(
    tracks_data: dict,
    interval,
    figsize=(14, 6),
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Plot multiple genomic tracks overlaid on the same plot.

    Args:
        tracks_data: Dictionary of track data {name: data_array}
        interval: Genomic interval for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each track
    for track_name, track_data in tracks_data.items():
        # Convert to numpy array if needed
        if hasattr(track_data, 'values'):
            y_data = track_data.values
        elif isinstance(track_data, list):
            y_data = np.array(track_data)
        else:
            y_data = track_data

        # Create x-axis (genomic position)
        x_data = np.linspace(interval.start, interval.end, len(y_data))

        # Plot track
        ax.plot(x_data, y_data, label=track_name, linewidth=1.5)

    # Add legend
    ax.legend(loc='best')

    # Set labels and title
    ax.set_xlabel(f'Genomic Position ({interval.chromosome})')
    ax.set_ylabel('Prediction Score')

    if title is None:
        title = f'Genomic Tracks: {interval.chromosome}:{interval.start}-{interval.end}'

    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def create_multi_panel_figure(
    outputs_list: list,
    titles: Optional[list] = None,
    figsize=(16, 10),
    save_path: Optional[Path] = None,
    main_title: Optional[str] = None
):
    """
    Create a multi-panel figure with multiple plots.

    Useful for creating publication-ready figures with multiple panels.

    Args:
        outputs_list: List of output objects to plot
        titles: Optional list of titles for each panel
        figsize: Overall figure size (width, height)
        save_path: Optional path to save the figure
        main_title: Optional main title for the entire figure

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None

    n_panels = len(outputs_list)

    # Calculate grid dimensions
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, (data, ax) in enumerate(zip(outputs_list, axes)):
        # Handle different data types
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # 1D array - line plot
                ax.plot(data, linewidth=1.5)
                ax.set_xlabel('Position')
                ax.set_ylabel('Value')
            elif data.ndim == 2:
                # 2D array - heatmap
                im = ax.imshow(data, aspect='auto', cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel('Position')
                ax.set_ylabel('Sample')
        else:
            # Fallback for unknown data types
            ax.text(0.5, 0.5, f'Panel {i+1}\nUnsupported data type', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)

        if titles and i < len(titles):
            ax.set_title(titles[i], fontweight='bold')
        else:
            ax.set_title(f'Panel {i+1}', fontweight='bold')

    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    # Add main title
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=0.995)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def save_figure(fig, filepath: Union[str, Path], formats: List[str] = None):
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib Figure object
        filepath: Base file path (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
    """
    if formats is None:
        formats = ['png', 'pdf']

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(output_path, bbox_inches='tight', format=fmt)
        logger.info(f"Figure saved to {output_path}")
