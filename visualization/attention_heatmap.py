"""Attention heatmap visualization (Fig.4).

Visualizes cross-modal attention patterns from the LLM agent,
showing how the model attends to flow features vs. payload tokens
during rule generation.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Custom blue-red colormap
_ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    'attention_cmap',
    ['#f7fbff', '#6baed6', '#2171b5', '#08306b'],
    N=256,
)


def extract_attention_weights(
    model_output,
    layer_idx: int = -1,
    head_indices: Optional[List[int]] = None,
    token_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Extract attention weights from LLM output.

    Args:
        model_output: Model generation output with attention weights
                      (HuggingFace output or vLLM attention dump).
        layer_idx: Which transformer layer to extract (-1 = last).
        head_indices: Specific attention heads (None = average all).
        token_range: (start, end) slice of token positions.

    Returns:
        (T, T) attention weight matrix or None if not available.
    """
    if hasattr(model_output, 'attentions') and model_output.attentions:
        attn = model_output.attentions[layer_idx]  # (batch, heads, seq, seq)
        weights = attn[0].detach().cpu().numpy()  # (heads, seq, seq)

        if head_indices is not None:
            weights = weights[head_indices]
        weights = weights.mean(axis=0)  # (seq, seq)

        if token_range is not None:
            s, e = token_range
            weights = weights[s:e, s:e]

        return weights

    # Fallback: generate synthetic attention pattern for visualization
    return None


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    token_labels: Optional[List[str]] = None,
    figsize: tuple = (7, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> plt.Figure:
    """Fig.4a: Single attention heatmap for one layer/head.

    Args:
        attention_weights: (T, T) attention matrix.
        token_labels: T labels for each token position.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.
        title: Optional sub-figure title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention_weights, cmap=_ATTENTION_CMAP, aspect='auto',
                   vmin=0, vmax=attention_weights.max())

    if token_labels and len(token_labels) <= 30:
        step = max(1, len(token_labels) // 15)
        ticks = list(range(0, len(token_labels), step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([token_labels[i] for i in ticks], rotation=45,
                           ha='right', fontsize=6)
        ax.set_yticklabels([token_labels[i] for i in ticks], fontsize=6)
    else:
        ax.tick_params(labelsize=7)

    ax.set_xlabel('Key Position', fontsize=9)
    ax.set_ylabel('Query Position', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Attention Weight', fontsize=8)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def plot_cross_modal_attention(
    flow_attention: np.ndarray,
    payload_attention: np.ndarray,
    flow_labels: Optional[List[str]] = None,
    payload_labels: Optional[List[str]] = None,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Fig.4b-c: Cross-modal attention comparison (flow vs. payload).

    Side-by-side heatmaps showing attention to flow features and payload tokens.

    Args:
        flow_attention: (T_f, T_f) attention to flow features.
        payload_attention: (T_p, T_p) attention to payload tokens.
        flow_labels: Labels for flow feature axes.
        payload_labels: Labels for payload token axes.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [1, 1.2]})

    im1 = ax1.imshow(flow_attention, cmap=_ATTENTION_CMAP, aspect='auto',
                     vmin=0, vmax=max(flow_attention.max(), payload_attention.max()))
    im2 = ax2.imshow(payload_attention, cmap=_ATTENTION_CMAP, aspect='auto',
                     vmin=0, vmax=max(flow_attention.max(), payload_attention.max()))

    ax1.set_title('(a) Flow Feature Attention', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Payload Token Attention', fontsize=10, fontweight='bold')

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Key Position', fontsize=8)
        ax.set_ylabel('Query Position', fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Attention Weight', fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig
