"""PCRE pattern generalization (Section III-E Step 5).

Converts specific payload matches into generalized regular expressions
that capture attack variants: encoding variations, case insensitivity,
whitespace flexibility, and comment obfuscation.
"""

import re
from typing import List, Optional

from ..utils.suricata_utils import SuricataRule


def generalize_sql_injection_pcre(content_patterns: List[str]) -> List[str]:
    """Generate generalized PCRE patterns for SQL injection detection.

    Handles common SQLi variants:
    - UNION SELECT with optional ALL keyword
    - Boolean-based (OR 1=1, AND 1=1)
    - Comment obfuscation (--, #, /**/)
    - String concatenation
    - URL encoding variants
    """
    pcre_list = []

    for pattern in content_patterns:
        pattern_lower = pattern.lower()

        if 'union' in pattern_lower and 'select' in pattern_lower:
            pcre_list.append(r"/UNION\s+(ALL\s+)?SELECT/i")
        elif 'select' in pattern_lower and 'from' in pattern_lower:
            pcre_list.append(r"/SELECT\s+.+\s+FROM\s+.+/i")
        elif "or" in pattern_lower and "1=1" in pattern_lower.replace(' ', ''):
            pcre_list.append(r"/\bOR\s+1\s*=\s*1\b/i")
        elif "and" in pattern_lower and "1=1" in pattern_lower.replace(' ', ''):
            pcre_list.append(r"/\bAND\s+1\s*=\s*1\b/i")
        elif "'" in pattern or '"' in pattern:
            pcre_list.append(r'/[\x27"]\s*(?:OR|AND)\s+/i')

    # Add comprehensive SQLi pattern if specific patterns found
    if pcre_list:
        pcre_list.append(
            r'''/(\%27|\'|--|#|UNION|OR\s+1=1|AND\s+1=1|SELECT.+FROM)/iU'''
        )

    return pcre_list


def generalize_xss_pcre(content_patterns: List[str]) -> List[str]:
    """Generate generalized PCRE for XSS detection."""
    pcre_list = []
    for pattern in content_patterns:
        pattern_lower = pattern.lower()
        if '<script' in pattern_lower:
            pcre_list.append(r"/<\s*script[^>]*>/i")
        if 'onerror' in pattern_lower or 'onload' in pattern_lower or 'onclick' in pattern_lower:
            pcre_list.append(r'/on(?:error|load|click|mouseover|focus)\s*=/i')
        if 'javascript:' in pattern_lower:
            pcre_list.append(r"/javascript\s*:/i")
        if 'alert(' in pattern_lower:
            pcre_list.append(r"/alert\s*\(/i")
        if '<img' in pattern_lower:
            pcre_list.append(r'/<\s*img\s+[^>]*on\w+\s*=/i')
    return pcre_list


def generalize_path_traversal_pcre(content_patterns: List[str]) -> List[str]:
    """Generate generalized PCRE for path traversal detection."""
    pcre_list = []
    for pattern in content_patterns:
        if '../' in pattern or '..\\' in pattern:
            pcre_list.append(r'/\.\.(?:/|\\)/')
        if '/etc/passwd' in pattern:
            pcre_list.append(r'/(?:/etc/(?:passwd|shadow|hosts)|C:\\Windows\\System32)/i')
        if 'cmd.exe' in pattern or '/bin/' in pattern:
            pcre_list.append(r'/(?:cmd\.exe|/bin/(?:sh|bash)|powershell)/i')
    return pcre_list


def generate_pcre_variants(rule: SuricataRule) -> SuricataRule:
    """Automatically expand a rule's PCRE patterns to cover attack variants.

    Examines existing content patterns and generates appropriate PCRE
    generalizations for common attack types.
    """
    msg_lower = rule.msg.lower()
    content_patterns = rule.content

    if any(kw in msg_lower for kw in ('sql', 'sqli', 'injection', 'select', 'union')):
        sqli_pcre = generalize_sql_injection_pcre(content_patterns)
        for p in sqli_pcre:
            if p not in rule.pcre:
                rule.pcre.append(p)

    if any(kw in msg_lower for kw in ('xss', 'cross-site', 'script', 'javascript')):
        xss_pcre = generalize_xss_pcre(content_patterns)
        for p in xss_pcre:
            if p not in rule.pcre:
                rule.pcre.append(p)

    if any(kw in msg_lower for kw in ('traversal', 'path', 'directory', '../')):
        traversal_pcre = generalize_path_traversal_pcre(content_patterns)
        for p in traversal_pcre:
            if p not in rule.pcre:
                rule.pcre.append(p)

    return rule


def simplify_pcre(pcre_list: List[str]) -> List[str]:
    """Remove redundant PCRE patterns and deduplicate."""
    seen = set()
    result = []
    for p in pcre_list:
        # Normalize
        normalized = p.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(p)
    return result
