"""Rule generalization module — specific → generalized rule transformation."""

from .ip_port_abstractor import abstract_ips, abstract_ports, abstract_rule
from .pcre_generalizer import generate_pcre_variants, simplify_pcre
from .sqli_patterns import expand_sqli_patterns, get_default_sqli_pcre, add_false_positive_mitigations
from .rule_transformer import RuleTransformer

__all__ = [
    'abstract_ips',
    'abstract_ports',
    'abstract_rule',
    'generate_pcre_variants',
    'simplify_pcre',
    'expand_sqli_patterns',
    'get_default_sqli_pcre',
    'add_false_positive_mitigations',
    'RuleTransformer',
]
