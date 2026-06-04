"""Parameter sensitivity curves (Fig.2).

Six sub-figures showing how each hyperparameter affects SV, DC, VC, and FPR:
  (a) RAG top-k        (k = 1, 3, 5, 7, 10)
  (b) Fusion weight    (alpha = 0.3, 0.5, 0.6, 0.7, 0.9)
  (c) Temperature      (tau = 0.03, 0.05, 0.07, 0.10, 0.15)
  (d) Max iterations   (N_max = 1, 2, 3, 4, 5)
  (e) LoRA rank        (r = 4, 8, 16, 32, 64)
  (f) Legend
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


@dataclass
class SensitivityResult:
    """Results for one sensitivity sweep."""
    param_name: str
    param_values: List[float]
    sv_mean: List[float]
    sv_std: List[float]
    dc_mean: List[float]
    dc_std: List[float]
    vc_mean: List[float]
    vc_std: List[float]
    fpr_mean: List[float]
    fpr_std: List[float]


def run_sensitivity_experiment(
    param_name: str,
    param_values: List[float],
    run_fn: Callable[[str, float], Dict[str, float]],
    n_seeds: int = 5,
) -> SensitivityResult:
    """Run a full sensitivity sweep for one parameter.

    Args:
        param_name: Name of the parameter being swept.
        param_values: List of parameter values to test.
        run_fn: Function taking (param_name, param_value) -> metrics dict
                with keys SV, DC, VC, FPR.
        n_seeds: Number of random seeds per value.

    Returns:
        SensitivityResult with means and stds computed across seeds.
    """
    sv_mean, sv_std = [], []
    dc_mean, dc_std = [], []
    vc_mean, vc_std = [], []
    fpr_mean, fpr_std = [], []

    for val in param_values:
        sv_vals, dc_vals, vc_vals, fpr_vals = [], [], [], []
        for seed in range(n_seeds):
            metrics = run_fn(param_name, val, seed=seed)
            sv_vals.append(metrics.get('SV', 0))
            dc_vals.append(metrics.get('DC', 0))
            vc_vals.append(metrics.get('VC', 0))
            fpr_vals.append(metrics.get('FPR', 0))

        sv_mean.append(np.mean(sv_vals))
        sv_std.append(np.std(sv_vals))
        dc_mean.append(np.mean(dc_vals))
        dc_std.append(np.std(dc_vals))
        vc_mean.append(np.mean(vc_vals))
        vc_std.append(np.std(vc_vals))
        fpr_mean.append(np.mean(fpr_vals))
        fpr_std.append(np.std(fpr_vals))

    return SensitivityResult(
        param_name=param_name,
        param_values=param_values,
        sv_mean=sv_mean, sv_std=sv_std,
        dc_mean=dc_mean, dc_std=dc_std,
        vc_mean=vc_mean, vc_std=vc_std,
        fpr_mean=fpr_mean, fpr_std=fpr_std,
    )


def _plot_one_sensitivity(
    ax: plt.Axes,
    result: SensitivityResult,
    metric: str,
    color: str,
    marker: str = 'o',
    xlabel: str = None,
):
    """Plot one metric curve on an axes with error bands."""
    mean_key = f'{metric.lower()}_mean'
    std_key = f'{metric.lower()}_std'

    x = np.arange(len(result.param_values))
    mean_vals = getattr(result, mean_key)
    std_vals = getattr(result, std_key)

    ax.plot(x, mean_vals, color=color, marker=marker, linewidth=1.5,
            markersize=5, label=metric.upper())
    ax.fill_between(x,
                    np.array(mean_vals) - np.array(std_vals),
                    np.array(mean_vals) + np.array(std_vals),
                    color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in result.param_values], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_sensitivity_curves(
    results: Dict[str, SensitivityResult],
    figsize: tuple = (14, 5.5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot Fig.2: full sensitivity analysis with 5 parameter panels + legend.

    Args:
        results: Dict mapping param_label -> SensitivityResult.
                 Expected keys: 'k (RAG)', 'alpha', 'tau', 'N_max', 'r (QLoRA)'.
        figsize: Figure size in inches.
        save_path: If provided, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure.
    """
    param_order = ['k (RAG)', 'alpha', 'tau', 'N_max', 'r (QLoRA)']
    colors = {'sv': '#2196F3', 'dc': '#4CAF50', 'vc': '#FF9800', 'fpr': '#F44336'}
    markers = {'sv': 'o', 'dc': 's', 'vc': '^', 'fpr': 'D'}

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5])

    for idx, param in enumerate(param_order):
        if param not in results:
            continue
        row, col = divmod(idx, 3)
        if col == 2:
            row, col = 0, 2

        ax = fig.add_subplot(gs[row, col])
        r = results[param]

        for metric_key in ['sv', 'dc', 'vc', 'fpr']:
            _plot_one_sensitivity(ax, r, metric_key, colors[metric_key],
                                  markers[metric_key])

        param_titles = {
            'k (RAG)': '(a) RAG Top-k',
            'alpha': '(b) Fusion Weight α',
            'tau': '(c) Temperature τ',
            'N_max': '(d) Max Iterations',
            'r (QLoRA)': '(e) LoRA Rank r',
        }
        ax.set_title(param_titles.get(param, param), fontsize=9, fontweight='bold')
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel('Value (%)', fontsize=8)

    # Legend panel (f)
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis('off')

    legend_elements = [
        plt.Line2D([0], [0], color=colors['sv'], marker=markers['sv'],
                   label='SV (Syntactic Validity)', linewidth=1.5, markersize=5),
        plt.Line2D([0], [0], color=colors['dc'], marker=markers['dc'],
                   label='DC (Detection Coverage)', linewidth=1.5, markersize=5),
        plt.Line2D([0], [0], color=colors['vc'], marker=markers['vc'],
                   label='VC (Variant Coverage)', linewidth=1.5, markersize=5),
        plt.Line2D([0], [0], color=colors['fpr'], marker=markers['fpr'],
                   label='FPR (False Positive Rate)', linewidth=1.5, markersize=5),
    ]
    ax_legend.legend(handles=legend_elements, loc='center left',
                     fontsize=7, frameon=True, fancybox=True,
                   title='Metrics', title_fontsize=8)

    ax_legend.text(0.5, 0.02, '(f)', transform=ax_legend.transAxes,
                    fontsize=9, fontweight='bold', ha='center')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig
