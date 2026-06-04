"""t-SNE embedding visualization (Fig.3).

Three separate figures:
  (a) Dual-embedding space with attack type coloring
  (b) Attack-type separation (Pcap2Rule vs baseline)
  (c) Exemplar proximity (retrieved rules vs query embeddings)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


ATTACK_COLORS = {
    'Benign': '#2ecc71',
    'Brute Force': '#e74c3c',
    'SQL Injection': '#3498db',
    'XSS': '#f39c12',
    'DDoS': '#9b59b6',
    'Infiltration': '#1abc9c',
    'Botnet': '#e67e22',
    'Port Scan': '#34495e',
}
ATTACK_MARKERS = {
    'Benign': 'o',
    'Brute Force': 's',
    'SQL Injection': '^',
    'XSS': 'D',
    'DDoS': 'v',
    'Infiltration': '<',
    'Botnet': '>',
    'Port Scan': 'p',
}


def _run_tsne(embeddings: np.ndarray, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    return TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                random_state=seed, init='pca', learning_rate='auto').fit_transform(embeddings)


def plot_tsne_dual_embedding(
    type_embeddings: np.ndarray,
    sig_embeddings: np.ndarray,
    labels: List[str],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Fig.3a: t-SNE of the dual-embedding space.

    Args:
        type_embeddings: (N, 768) attack-type embeddings.
        sig_embeddings: (N, 384) signature embeddings.
        labels: N attack-type labels for coloring.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.

    Returns:
        matplotlib Figure.
    """
    combined = np.concatenate([type_embeddings, sig_embeddings], axis=1)
    coords = _run_tsne(combined)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels), key=lambda x: list(ATTACK_COLORS.keys()).index(x)
                           if x in ATTACK_COLORS else 99)

    for label in unique_labels:
        mask = [l == label for l in labels]
        color = ATTACK_COLORS.get(label, '#95a5a6')
        marker = ATTACK_MARKERS.get(label, 'o')
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, marker=marker,
                   label=label, s=30, alpha=0.75, edgecolors='white',
                   linewidth=0.3)

    ax.legend(loc='best', fontsize=7, frameon=True, fancybox=True,
              ncol=2 if len(unique_labels) > 4 else 1)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=9)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def plot_tsne_attack_types(
    pcap2rule_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    labels: List[str],
    baseline_name: str = 'Baseline',
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Fig.3b: Side-by-side comparison of Pcap2Rule vs baseline embedding quality.

    Intra-class compactness values are returned for use in figure captions.

    Args:
        pcap2rule_embeddings: (N, D) Pcap2Rule embeddings.
        baseline_embeddings: (N, D) baseline embeddings.
        labels: N class labels.
        baseline_name: Name of the baseline method.
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.

    Returns:
        matplotlib Figure.
    """
    pcap2r_coords = _run_tsne(pcap2rule_embeddings)
    baseline_coords = _run_tsne(baseline_embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = [l == label for l in labels]
        color = ATTACK_COLORS.get(label, '#95a5a6')
        ax1.scatter(pcap2r_coords[mask, 0], pcap2r_coords[mask, 1],
                    c=color, s=25, alpha=0.7, edgecolors='white', linewidth=0.2)
        ax2.scatter(baseline_coords[mask, 0], baseline_coords[mask, 1],
                    c=color, s=25, alpha=0.7, edgecolors='white', linewidth=0.2)

    # Compute intra-class compactness
    pcap2r_compact = _intra_class_compactness(pcap2rule_embeddings, labels)
    base_compact = _intra_class_compactness(baseline_embeddings, labels)

    ax1.set_title(f'Pcap2Rule (intra: {pcap2r_compact:.2f})', fontsize=10, fontweight='bold')
    ax2.set_title(f'{baseline_name} (intra: {base_compact:.2f})', fontsize=10, fontweight='bold')

    for ax in (ax1, ax2):
        ax.set_xlabel('t-SNE 1', fontsize=8)
        ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def plot_tsne_with_exemplars(
    query_embeddings: np.ndarray,
    exemplar_embeddings: np.ndarray,
    query_labels: List[str],
    exemplar_labels: List[str],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Fig.3c: t-SNE showing queries and their retrieved exemplars.

    Draws connecting lines between each query and its top retrieved exemplar.

    Args:
        query_embeddings: (Q, D) query embeddings.
        exemplar_embeddings: (K, D) retrieved exemplar embeddings.
        query_labels: Q labels.
        exemplar_labels: K labels (corresponding to each query's exemplar).
        figsize: Figure size.
        save_path: Optional save path.
        dpi: Output resolution.

    Returns:
        matplotlib Figure.
    """
    all_emb = np.concatenate([query_embeddings, exemplar_embeddings], axis=0)
    n_q = len(query_embeddings)
    coords = _run_tsne(all_emb)

    q_coords = coords[:n_q]
    e_coords = coords[n_q:]

    fig, ax = plt.subplots(figsize=figsize)

    # Draw connecting lines
    for i in range(n_q):
        ax.plot([q_coords[i, 0], e_coords[i, 0]],
                [q_coords[i, 1], e_coords[i, 1]],
                'k-', alpha=0.15, linewidth=0.5)

    # Queries
    unique_labels = sorted(set(query_labels))
    for label in unique_labels:
        mask = [l == label for l in query_labels]
        color = ATTACK_COLORS.get(label, '#95a5a6')
        ax.scatter(q_coords[mask, 0], q_coords[mask, 1],
                   c=color, marker='o', s=40, alpha=0.8,
                   edgecolors='black', linewidth=0.5, label=label)

    # Exemplars (stars)
    ax.scatter(e_coords[:, 0], e_coords[:, 1],
               c='#e74c3c', marker='*', s=80, alpha=0.9,
               edgecolors='black', linewidth=0.3, label='Exemplars')

    ax.legend(loc='best', fontsize=7, frameon=True, fancybox=True)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=9)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def _intra_class_compactness(embeddings: np.ndarray, labels: List[str]) -> float:
    """Compute mean pairwise cosine distance within each class, averaged across classes."""
    from sklearn.metrics.pairwise import cosine_distances

    unique_labels = set(labels)
    mean_distances = []
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        if mask.sum() < 2:
            continue
        class_emb = embeddings[mask]
        dists = cosine_distances(class_emb)
        triu_idx = np.triu_indices_from(dists, k=1)
        mean_distances.append(dists[triu_idx].mean())
    return np.mean(mean_distances) if mean_distances else 0.0
