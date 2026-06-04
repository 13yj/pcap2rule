"""Confusion matrix visualization (Fig.5).

Two plots:
  (a) Attack classification confusion matrix (XGBoost)
  (b) Per-attack-type detection coverage bar chart (Pcap2Rule vs baselines)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    'confusion_cmap',
    ['#ffffff', '#deebf7', '#9ecae1', '#4292c6', '#08519c'],
    N=256,
)


def compute_confusion_matrix_data(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    normalize: str = 'true',
) -> Tuple[np.ndarray, List[str]]:
    """Compute confusion matrix for attack type classification.

    Args:
        y_true: Ground truth attack type labels.
        y_pred: Predicted attack type labels.
        labels: Ordered list of class labels.
        normalize: 'true' for row-normalized, 'all' for global, None for counts.

    Returns:
        (cm, labels) tuple.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm, labels


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    figsize: tuple = (7, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: Optional[str] = None,
    fmt: str = '.2f',
) -> plt.Figure:
    """Fig.5a: Confusion matrix heatmap.

    Args:
        cm: (C, C) confusion matrix (row-normalized).
        labels: C class names.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.
        title: Optional title.
        fmt: Value format string.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, cmap=CONFUSION_CMAP, aspect='auto', vmin=0, vmax=1.0)

    # Annotate cells
    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, format(val, fmt), ha='center', va='center',
                    fontsize=8 if n <= 8 else 6, color=color,
                    fontweight='bold' if val > 0.5 else 'normal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Fraction', fontsize=8)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def plot_per_attack_comparison(
    attack_types: List[str],
    pcap2rule_dc: List[float],
    baseline_dc: Dict[str, List[float]],
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Fig.5b: Per-attack-type detection coverage grouped bar chart.

    Args:
        attack_types: Ordered list of attack type names.
        pcap2rule_dc: DC values (%) for Pcap2Rule for each attack type.
        baseline_dc: Dict mapping baseline name -> DC list for each attack type.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.

    Returns:
        matplotlib Figure.
    """
    n_attacks = len(attack_types)
    n_methods = 1 + len(baseline_dc)
    width = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_attacks)

    # Pcap2Rule bars
    bars1 = ax.bar(x - (n_methods - 1) * width / 2, pcap2rule_dc, width,
                   color='#2196F3', edgecolor='white', linewidth=0.5,
                   label='Pcap2Rule', zorder=3)

    # Baseline bars
    baseline_colors = ['#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#795548']
    for idx, (name, dc_vals) in enumerate(baseline_dc.items()):
        color = baseline_colors[idx % len(baseline_colors)]
        offset = x - (n_methods - 1) * width / 2 + (idx + 1) * width
        ax.bar(offset, dc_vals, width, color=color, edgecolor='white',
               linewidth=0.5, label=name, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Detection Coverage (%)', fontsize=9)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.legend(fontsize=7, frameon=True, fancybox=True)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}',
                ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                color='#2196F3')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig
