"""SQL injection pattern expansion (Section III-E Step 5, Section IV-L case study).

Maps specific SQL injection payloads to broad detection patterns.
Reference: the paper's case study where a specific UNION SELECT rule
is generalized to detect URL-encoded variants, comment obfuscation,
tautologies, and time-based injection.
"""

from typing import List, Dict


# SQLi pattern taxonomy
SQLI_PATTERNS = {
    'union_based': [
        r'UNION\s+(ALL\s+)?SELECT',
        r'UNION/\*\*/\s*SELECT',  # Comment obfuscation
    ],
    'tautology': [
        r"\bOR\s+['\"]?\s*\d+\s*=\s*\d+",
        r"\bAND\s+['\"]?\s*\d+\s*=\s*\d+",
        r"'\s*=\s*'",
        r'"\s*=\s*"',
    ],
    'comment_obfuscation': [
        r'/\*\*!/',           # MySQL conditional comment
        r'/\*!.+\*/',          # MySQL version-specific
        r'--[\s-]',            # SQL comment
        r'#',                  # MySQL comment
    ],
    'stacked_queries': [
        r';\s*(?:DROP|DELETE|INSERT|UPDATE|EXEC|SELECT|ALTER)',
    ],
    'time_based': [
        r"(?:WAITFOR\s+DELAY|SLEEP\s*\(|BENCHMARK\s*\()",
        r"pg_sleep\s*\(",
    ],
    'information_schema': [
        r'information_schema\.(?:TABLES|COLUMNS|SCHEMATA)',
    ],
    'string_concat': [
        r"(?:CONCAT\s*\(|GROUP_CONCAT\s*\()",
        r"\|\|",  # Oracle/PostgreSQL concat
    ],
    'url_encoding': [
        r'%27',    # '
        r'%22',    # "
        r'%23',    # #
        r'%2D%2D', # --
        r'%3B',    # ;
        r'%20',    # space
    ],
}


def expand_sqli_patterns(payload_text: str) -> List[str]:
    """Expand observed SQLi patterns into generalized detection patterns.

    Given a specific SQL injection payload (e.g., "GET /products.php?id=1'
    UNION SELECT username,password FROM users--"), returns the set of
    generalized PCRE patterns needed for variant-resistant detection.

    Args:
        payload_text: The observed SQL injection payload text.

    Returns:
        List of generalized PCRE pattern strings.
    """
    payload_lower = payload_text.lower()
    patterns = []

    # Check which categories are relevant
    has_union = 'union' in payload_lower and 'select' in payload_lower
    has_tautology = any(
        kw in payload_lower for kw in ("'='", '1=1', "'or'", "'and'")
    )
    has_comment = any(
        kw in payload_lower for kw in ('--', '#', '/**/')
    )
    has_stacked = ';' in payload_lower and any(
        kw in payload_lower for kw in ('drop', 'delete', 'insert', 'exec')
    )
    has_time = any(
        kw in payload_lower for kw in ('sleep', 'waitfor', 'benchmark', 'pg_sleep')
    )
    has_info_schema = 'information_schema' in payload_lower
    has_concat = 'concat' in payload_lower or '||' in payload_lower
    has_encoding = any(
        kw in payload_lower for kw in ('%27', '%22', '%23', '%20')
    )

    # Generate patterns based on observed categories
    if has_union:
        patterns.extend(SQLI_PATTERNS['union_based'])
        patterns.append(
            r"/(?:SELECT|UNION).+(?:FROM|INTO\s+(?:OUTFILE|DUMPFILE))/i"
        )

    if has_tautology:
        patterns.extend(SQLI_PATTERNS['tautology'])
        patterns.append(r"/['\"]\s*(?:OR|AND)\s+['\"]?\s*['\"]/i")

    if has_comment:
        patterns.extend(SQLI_PATTERNS['comment_obfuscation'])

    if has_stacked:
        patterns.extend(SQLI_PATTERNS['stacked_queries'])

    if has_time:
        patterns.extend(SQLI_PATTERNS['time_based'])

    if has_info_schema:
        patterns.extend(SQLI_PATTERNS['information_schema'])

    if has_concat:
        patterns.extend(SQLI_PATTERNS['string_concat'])

    if has_encoding:
        patterns.extend(SQLI_PATTERNS['url_encoding'])

    # Default: comprehensive SQLi pattern (used in paper case study)
    if not patterns:
        patterns.append(
            r"/(\%27|\'|--|#|UNION|OR\s+1=1|AND\s+1=1|SELECT.+FROM)/iU"
        )

    return patterns


def get_default_sqli_pcre() -> str:
    """Return the default comprehensive SQLi PCRE from the paper's case study.

    This is the generalized rule pattern from Section IV-L:
        pcre:"/(\\%27|\\'|--|#|UNION|OR\\s+1=1|AND\\s+1=1|SELECT.+FROM)/iU"
    """
    return r"/(\%27|\'|--|#|UNION|OR\s+1=1|AND\s+1=1|SELECT.+FROM)/iU"


def add_false_positive_mitigations(rule_patterns: List[str]) -> List[str]:
    """Add additional constraints to reduce false positives.

    Returns modified patterns with word boundaries, escaping,
    and scope restrictions.
    """
    mitigated = []
    for p in rule_patterns:
        # Add word boundaries to literal keywords
        p = p.replace(r"UNION\s", r"\bUNION\s")
        p = p.replace(r"SELECT\s", r"\bSELECT\s")
        p = p.replace(r"FROM\s", r"\bFROM\s")

        # Add case insensitivity if not present
        if not p.endswith('/i') and not p.endswith('/iU'):
            p = p.rstrip('/') + '/i'

        mitigated.append(p)
    return mitigated
