"""Human evaluation data preparation (Section IV-M).

Prepares rule evaluation forms and computes inter-rater agreement
(Fleiss' kappa) for the human expert evaluation described in the paper.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def compute_fleiss_kappa(ratings: List[List[int]]) -> float:
    """Compute Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: N × R matrix where N = number of subjects (rules),
                 R = number of raters, and values are category indices (0, 1, 2).

    Returns:
        Fleiss' kappa value ∈ [-1, 1].
    """
    ratings = np.array(ratings)
    n = ratings.shape[0]   # number of subjects (rules)
    r = ratings.shape[1]   # number of raters
    n_categories = int(ratings.max()) + 1

    # Proportion of all assignments to each category
    p_j = np.zeros(n_categories)
    for j in range(n_categories):
        p_j[j] = (ratings == j).sum() / (n * r)

    # Per-subject agreement
    P_i = np.zeros(n)
    for i in range(n):
        counts = Counter(ratings[i])
        P_i[i] = (sum(c * (c - 1) for c in counts.values())) / (r * (r - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_bar == P_e and P_bar == 1.0:
        return 1.0
    if P_e == 1.0:
        return 0.0

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return kappa


def compute_approval_rates(
    ratings: List[List[int]],
    approval_threshold: int = 1,  # 0=Reject, 1=Needs Tuning, 2=Deployable
) -> Dict[str, float]:
    """Compute analyst approval rates from ratings.

    Args:
        ratings: N × R matrix as in compute_fleiss_kappa.
        approval_threshold: Ratings >= this value count as "approved".

    Returns:
        Dict with 'deployable', 'needs_tuning', 'reject', 'approval' rates.
    """
    ratings = np.array(ratings)
    total = ratings.size

    deployable = (ratings == 2).sum() / total * 100
    needs_tuning = (ratings == 1).sum() / total * 100
    reject = (ratings == 0).sum() / total * 100
    approval = ((ratings >= approval_threshold).sum() / total) * 100

    return {
        'deployable': deployable,
        'needs_tuning': needs_tuning,
        'reject': reject,
        'approval': approval,
    }


def split_by_rule_type(
    ratings_specific: List[List[int]],
    ratings_generalized: List[List[int]],
    approval_threshold: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Compute approval rates separately for specific and generalized rules.

    Returns:
        Dict with 'specific' and 'generalized' keys, each containing
        the approval rates dict from compute_approval_rates().
    """
    return {
        'specific': compute_approval_rates(ratings_specific, approval_threshold),
        'generalized': compute_approval_rates(ratings_generalized, approval_threshold),
    }
