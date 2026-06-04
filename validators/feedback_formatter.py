"""Feedback formatter — converts validator outputs to structured feedback text.

Formats raw validator error/warning data into LLM-friendly structured feedback
that the agent can reason about and act upon during refinement iterations.
"""

from typing import List, Dict, Any


def format_syntax_feedback(errors: List[str]) -> str:
    """Format syntax validation errors into actionable feedback."""
    if not errors:
        return ""
    lines = ["[SYNTAX VALIDATION FAILED]", ""]
    for err in errors:
        lines.append(f"  - {err}")
    lines.append("")
    lines.append("Please correct ALL syntax errors before re-submitting.")
    return "\n".join(lines)


def format_semantic_feedback(score: float, threshold: float, issues: List[str]) -> str:
    """Format semantic validation results."""
    lines = [
        f"[SEMANTIC VALIDATION] Score: {score:.3f} (threshold: {threshold:.2f})",
        "",
    ]
    if score >= threshold:
        lines.append("  Semantic alignment: PASSED")
    else:
        lines.append("  Semantic alignment: FAILED")
        if issues:
            lines.append("  Issues detected:")
            for issue in issues:
                lines.append(f"    - {issue}")
        lines.append("")
        lines.append("Suggestions for improvement:")
        lines.append("  - Ensure content keywords appear in the observed traffic payload")
        lines.append("  - Match protocol to the actual application protocol observed")
        lines.append("  - Add structural elements (flow direction, classtype)")
    return "\n".join(lines)


def format_performance_feedback(fpr: float, threshold: float, details: str) -> str:
    """Format performance (FPR) validation results."""
    lines = [
        f"[PERFORMANCE VALIDATION] FPR: {fpr:.2%} (threshold: {threshold:.1%})",
        "",
    ]
    if fpr <= threshold:
        lines.append("  False positive check: PASSED")
    else:
        lines.append("  False positive check: FAILED")
        if details:
            lines.append(f"  {details}")
        lines.append("")
        lines.append("Recommended actions:")
        lines.append("  1. Add more specific content matches")
        lines.append("  2. Restrict matching scope (http_uri, http_header)")
        lines.append("  3. Add threshold with count/seconds limits")
        lines.append("  4. Use pcre with word boundaries (\\b) instead of raw content")
    return "\n".join(lines)


def combine_feedback(components: Dict[str, str]) -> str:
    """Combine all feedback components into a single structured message.

    Args:
        components: Dict mapping validator name to its feedback string.

    Returns:
        Combined feedback string.
    """
    parts = ["=" * 50, "VALIDATION FEEDBACK SUMMARY", "=" * 50, ""]

    for name, feedback in components.items():
        if feedback:
            parts.append(feedback)
            parts.append("")

    # Count failures
    failures = [k for k, v in components.items() if v and 'FAILED' in v]
    if failures:
        parts.append(f"{len(failures)} validator(s) failed.")
    else:
        parts.append("All validators passed.")

    return "\n".join(parts)
