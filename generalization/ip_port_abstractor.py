"""IP and port abstraction — replaces concrete addresses/ports with variables.

Section III-E Step 5: Converts concrete IPs → $EXTERNAL_NET/$HOME_NET
and specific ports → 'any' or known service ports.
"""

import re
import ipaddress
from typing import Tuple

from ..utils.suricata_utils import SuricataRule


# Known service ports
SERVICE_PORTS = {
    80: '$HTTP_PORTS',
    443: '$HTTP_PORTS',
    8080: '$HTTP_PORTS',
    8443: '$HTTP_PORTS',
    53: '53',
    22: '22',
    21: '21',
    25: '25',
    110: '110',
    143: '143',
    3306: '3306',
    3389: '3389',
}

# Private IP ranges
PRIVATE_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
]


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in the private range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        for net in PRIVATE_RANGES:
            if addr in net:
                return True
        return False
    except ValueError:
        return False


def abstract_ips(rule: SuricataRule) -> SuricataRule:
    """Replace concrete IPs with Suricata variables.

    - Internal (private) IPs → $HOME_NET
    - External IPs → $EXTERNAL_NET
    """
    if not rule.src_ip or rule.src_ip == 'any':
        pass
    elif rule.src_ip in ('$EXTERNAL_NET', '$HOME_NET', '$HTTP_SERVERS',
                         '$SQL_SERVERS', '$DNS_SERVERS'):
        pass  # already abstracted
    elif is_private_ip(rule.src_ip):
        rule.src_ip = '$HOME_NET'
    else:
        rule.src_ip = '$EXTERNAL_NET'

    if not rule.dst_ip or rule.dst_ip == 'any':
        pass
    elif rule.dst_ip in ('$EXTERNAL_NET', '$HOME_NET'):
        pass
    elif is_private_ip(rule.dst_ip):
        rule.dst_ip = '$HOME_NET'
    else:
        rule.dst_ip = '$EXTERNAL_NET'

    return rule


def abstract_ports(rule: SuricataRule) -> SuricataRule:
    """Abstract specific ports to 'any' or known service variables.

    Only preserves ports when they are essential for detection
    (e.g., HTTP attacks on port 80, DNS on 53).
    """
    try:
        src_port = int(rule.src_port)
        rule.src_port = 'any'  # source ports are ephemeral — always abstract
    except (ValueError, TypeError):
        pass  # already abstracted or variable

    try:
        dst_port = int(rule.dst_port)
        if dst_port in SERVICE_PORTS and dst_port in (80, 443, 8080, 8443):
            # HTTP/HTTPS — keep as variable for clarity
            rule.dst_port = '$HTTP_PORTS'
        elif dst_port == 53:
            rule.dst_port = '53'  # DNS is specific enough to keep
        else:
            rule.dst_port = 'any'
    except (ValueError, TypeError):
        pass

    return rule


def abstract_rule(rule: SuricataRule) -> SuricataRule:
    """Apply complete IP/port abstraction to a rule."""
    rule = abstract_ips(rule)
    rule = abstract_ports(rule)
    return rule
