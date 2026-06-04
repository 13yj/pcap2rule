"""Adversarial robustness evaluation — 4 perturbation types (Section IV-K).

Generates four categories of input perturbations:
1. Payload Obfuscation (PO)
2. Timing Perturbation (TP)
3. Flow Fragmentation (FF)
4. Protocol Mimicry (PM)
"""

import re
import random
import copy
import numpy as np
from typing import List, Tuple, Optional, Dict


def perturb_payload_obfuscation(
    payload_text: str,
    url_encode_pct: float = 0.5,
    case_randomize: bool = True,
    whitespace_insert: bool = True,
    seed: int = 42,
) -> str:
    """Apply payload obfuscation perturbations.

    - URL-encode special characters
    - Randomize case of HTTP method keywords
    - Insert whitespace in SQL keywords
    """
    rng = random.Random(seed)
    text = payload_text

    # URL-encode special characters
    if url_encode_pct > 0:
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch in "'\"#; " and rng.random() < url_encode_pct:
                chars[i] = f"%{ord(ch):02X}"
        text = ''.join(chars)

    # Case randomization of HTTP methods
    if case_randomize:
        for method in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']:
            if method in text:
                randomized = ''.join(
                    c.upper() if rng.random() < 0.5 else c.lower()
                    for c in method
                )
                text = text.replace(method, randomized, 1)

    # Whitespace insertion in SQL keywords
    if whitespace_insert:
        sql_keywords = ['UNION', 'SELECT', 'FROM', 'WHERE', 'AND', 'OR']
        for kw in sql_keywords:
            if kw in text.upper():
                # Insert extra spaces/tabs
                variant = kw[0] + ' ' + kw[1:]
                text = re.sub(kw, variant, text, count=1, flags=re.IGNORECASE)

    return text


def perturb_timing(
    flow_features: np.ndarray,
    jitter_pct: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Apply timing perturbation — add ±50% jitter to temporal features.

    Args:
        flow_features: (4, 38) aggregated flow features.
        jitter_pct: Maximum relative timing perturbation.

    Returns:
        Perturbed flow_features.
    """
    rng = np.random.RandomState(seed)
    perturbed = flow_features.copy()

    # Temporal dimensions (indices 0-9)
    for dim in range(10):
        jitter = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        perturbed[:, dim] *= jitter

    return perturbed


def perturb_fragmentation(
    features_shape: Tuple[int, int],
    max_segments: int = 4,
    seed: int = 42,
) -> Dict[str, object]:
    """Simulate flow fragmentation by splitting payload across segments.

    Returns metadata describing the fragmentation, which the payload
    extractor uses to reconstruct split payloads.
    """
    rng = random.Random(seed)
    n_segments = rng.randint(2, max_segments)
    out_of_order = rng.random() < 0.5

    return {
        'fragmented': True,
        'n_segments': n_segments,
        'out_of_order': out_of_order,
    }


def perturb_protocol_mimicry(
    payload_text: str,
    mimic_browser_ua: bool = True,
    seed: int = 42,
) -> str:
    """Modify HTTP headers to mimic benign browser traffic.

    Replaces suspicious User-Agent strings with common browser UAs
    while preserving the attack payload.
    """
    rng = random.Random(seed)

    BENIGN_UAS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]

    BENIGN_ACCEPT = [
        'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    ]

    text = payload_text

    if mimic_browser_ua:
        # Replace User-Agent
        ua_match = re.search(r'User-Agent:\s*(.+)', text, re.IGNORECASE)
        if ua_match:
            benign_ua = rng.choice(BENIGN_UAS)
            text = text[:ua_match.start(1)] + benign_ua + text[ua_match.end(1):]

        # Replace Accept header
        accept_match = re.search(r'Accept:\s*(.+)', text, re.IGNORECASE)
        if accept_match:
            benign_accept = rng.choice(BENIGN_ACCEPT)
            text = (text[:accept_match.start(1)] + benign_accept +
                    text[accept_match.end(1):])

    return text


def apply_all_perturbations(
    flow_features: np.ndarray,
    payload_text: str,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """Apply all four perturbation types simultaneously."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # PO
    payload_text = perturb_payload_obfuscation(
        payload_text, url_encode_pct=0.5, case_randomize=True,
        whitespace_insert=True, seed=seed,
    )
    # TP
    flow_features = perturb_timing(flow_features, jitter_pct=0.5, seed=seed + 1)
    # PM
    payload_text = perturb_protocol_mimicry(payload_text, seed=seed + 2)
    # FF (metadata only — actual fragmentation handled by extractor)

    return flow_features, payload_text


def evaluate_robustness(
    original_dc: float,
    perturbed_dcs: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """Compute DC drop and recovery rate for each perturbation type.

    Args:
        original_dc: DC on clean test data.
        perturbed_dcs: Dict mapping perturbation type → DC on perturbed data.

    Returns:
        Dict mapping perturbation → (dc_drop, recovery_rate) tuple.
    """
    results = {}
    for pert_type, pert_dc in perturbed_dcs.items():
        dc_drop = pert_dc - original_dc  # negative or positive
        recovery_rate = (pert_dc / original_dc) * 100.0 if original_dc > 0 else 0.0
        results[pert_type] = (dc_drop, recovery_rate)
    return results
