"""Visualization module — paper figures (Fig.2 through Fig.5)."""

from .sensitivity_curves import (
    plot_sensitivity_curves,
    SensitivityResult,
    run_sensitivity_experiment,
)
from .tsne_plot import (
    plot_tsne_dual_embedding,
    plot_tsne_attack_types,
    plot_tsne_with_exemplars,
)
from .attention_heatmap import (
    plot_attention_heatmap,
    plot_cross_modal_attention,
    extract_attention_weights,
)
from .confusion_matrix import (
    plot_confusion_matrix,
    plot_per_attack_comparison,
    compute_confusion_matrix_data,
)

__all__ = [
    'plot_sensitivity_curves',
    'SensitivityResult',
    'run_sensitivity_experiment',
    'plot_tsne_dual_embedding',
    'plot_tsne_attack_types',
    'plot_tsne_with_exemplars',
    'plot_attention_heatmap',
    'plot_cross_modal_attention',
    'extract_attention_weights',
    'plot_confusion_matrix',
    'plot_per_attack_comparison',
    'compute_confusion_matrix_data',
]
