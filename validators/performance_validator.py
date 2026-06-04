"""Stage 3: Performance validator — False Positive Rate evaluation (Section III-F.3, Eq.6).

Deploys generated rules on a held-out benign traffic sample and measures FPR.
If FPR > 5%, returns structured feedback including matched benign patterns
to help the agent add restrictive conditions.
"""

import re
import subprocess
import tempfile
import os
from typing import List, Tuple, Optional
import numpy as np

from ..utils.suricata_utils import SuricataRule
from ..utils.logging import get_logger


class PerformanceValidator:
    """Evaluate rule false positive rate on benign traffic (Eq.6).

    Writes rule to a temp file, runs suricata against a benign PCAP set,
    and counts matches. Returns FPR and structured feedback.

    Args:
        benign_pcaps: List of paths to benign PCAP files.
        suricata_binary: Path to suricata executable.
        fpr_threshold: Maximum acceptable FPR (default 5%).
        timeout: Timeout per suricata invocation.
    """

    def __init__(
        self,
        benign_pcaps: List[str],
        suricata_binary: str = "suricata",
        fpr_threshold: float = 0.05,
        timeout: int = 60,
    ):
        self.benign_pcaps = benign_pcaps
        self.suricata_binary = suricata_binary
        self.fpr_threshold = fpr_threshold
        self.timeout = timeout
        self.logger = get_logger("pcap2rule.performance")

    def check(self, rule: SuricataRule) -> Tuple[float, str]:
        """Measure FPR of a rule on benign traffic.

        Args:
            rule: SuricataRule to evaluate.

        Returns:
            (fpr, feedback) tuple. fpr ∈ [0, 1].
            Feedback is empty if fpr <= threshold.
        """
        if rule is None:
            return 1.0, "[FPR ERROR] Null rule — cannot evaluate performance."

        if not self.benign_pcaps:
            self.logger.warning("No benign PCAPs available. Skipping FPR check.")
            return 0.0, ""

        total = len(self.benign_pcaps)
        if total == 0:
            return 0.0, ""

        matches = self._count_matches(rule)

        # Also do a structural heuristic check
        heuristic_fpr = self._heuristic_fpr(rule)

        # Combine suricata results with heuristic
        if total > 0:
            fpr = (matches / total) * 0.7 + heuristic_fpr * 0.3
        else:
            fpr = heuristic_fpr

        if fpr <= self.fpr_threshold:
            return fpr, ""

        feedback = self._build_fpr_feedback(rule, fpr, matches, total)
        return fpr, feedback

    def _count_matches(self, rule: SuricataRule) -> int:
        """Count how many benign PCAPs trigger the rule."""
        # Write rule to temp file
        rule_file = os.path.join(
            tempfile.gettempdir(),
            f"_pcap2rule_fpr_{os.getpid()}.rules",
        )
        matches = 0

        try:
            with open(rule_file, 'w') as f:
                f.write(rule.to_string() + "\n")

            for pcap in self.benign_pcaps[:20]:  # max 20 for performance
                cmd = [
                    self.suricata_binary, '-r', pcap,
                    '-S', rule_file,
                    '-l', tempfile.gettempdir(),
                    '--runmode', 'single',
                ]
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                    # Check fast.log or eve.json for alerts
                    output = result.stdout + result.stderr
                    if 'alerts' in output.lower():
                        # Count alerts in output
                        alert_count = len(re.findall(r'Alerts?:?\s*(\d+)', output))
                        if alert_count > 0:
                            matches += 1
                except subprocess.TimeoutExpired:
                    matches += 1  # timeout = likely match
                except FileNotFoundError:
                    self.logger.warning("Suricata not found. Skipping live FPR check.")
                    return 0

        finally:
            try:
                os.remove(rule_file)
            except OSError:
                pass

        return matches

    def _heuristic_fpr(self, rule: SuricataRule) -> float:
        """Estimate FPR from rule structural properties without running suricata.

        Returns an estimated FPR ∈ [0, 1] based on rule broadness.
        """
        risk = 0.0

        # Broad content matches increase FPR risk
        for content in rule.content:
            if len(content) <= 3:
                risk += 0.15
            elif len(content) <= 6:
                risk += 0.08
            else:
                risk += 0.02

        # Broad pcre patterns
        for pcre in rule.pcre:
            if '.*' in pcre:
                risk += 0.1
            if '.+' in pcre:
                risk += 0.05

        # No flow constraint
        if not rule.flow:
            risk += 0.1

        # using 'any' for both ports
        if rule.src_port == 'any' and rule.dst_port == 'any':
            risk += 0.05

        # Using $EXTERNAL_NET (broad)
        if '$EXTERNAL_NET' in rule.src_ip or '$EXTERNAL_NET' in rule.dst_ip:
            risk += 0.03

        # No threshold
        if not rule.threshold:
            risk += 0.02

        return min(risk, 1.0)

    def _build_fpr_feedback(
        self,
        rule: SuricataRule,
        fpr: float,
        matches: int,
        total: int,
    ) -> str:
        """Build structured FPR feedback for the agent."""
        lines = [
            f"[FPR WARNING] False positive rate {fpr:.1%} > threshold "
            f"{self.fpr_threshold:.0%}:",
        ]

        if total > 0:
            lines.append(
                f"  Matched {matches}/{total} benign traffic samples."
            )
        else:
            lines.append(
                f"  Estimated FPR from rule structure: {fpr:.1%}."
            )

        lines.append("  Suggestions to reduce FPR:")
        lines.append("    - Add more specific content keywords to narrow matching")
        lines.append("    - Add 'flow' direction constraint (to_server / to_client)")
        lines.append(
            "    - Use 'http_uri' or 'http_header' to restrict search scope"
        )
        lines.append(
            "    - Add 'threshold: type threshold, track by_src, count N, seconds M'"
        )
        lines.append(
            "    - Add 'pcre' with stricter anchoring (^, $, \\b word boundaries)"
        )

        return "\n".join(lines)
