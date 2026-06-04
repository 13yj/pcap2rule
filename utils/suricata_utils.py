"""Suricata rule parsing, validation, and generation utilities.

Provides rule parsing into structured components and programmatic construction
of Suricata rules with correct syntax.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SuricataRule:
    """Structured representation of a Suricata rule."""

    action: str = "alert"
    protocol: str = "http"
    src_ip: str = "$EXTERNAL_NET"
    src_port: str = "any"
    direction: str = "->"
    dst_ip: str = "$HOME_NET"
    dst_port: str = "any"
    msg: str = ""
    flow: Optional[str] = None
    content: List[str] = field(default_factory=list)
    pcre: List[str] = field(default_factory=list)
    nocase: bool = False
    http_uri: bool = False
    threshold: Optional[Dict[str, Any]] = None
    classtype: str = "trojan-activity"
    sid: int = 100000
    rev: int = 1
    reference: Optional[str] = None
    metadata: List[str] = field(default_factory=list)
    raw: str = ""

    def to_string(self) -> str:
        """Serialize the rule to a valid Suricata rule string."""
        parts = [f'{self.action} {self.protocol} {self.src_ip} {self.src_port} '
                 f'{self.direction} {self.dst_ip} {self.dst_port}']

        options = []
        if self.msg:
            msg_escaped = self.msg.replace('"', '\\"')
            options.append(f'msg:"{msg_escaped}"')
        if self.flow:
            options.append(f'flow:{self.flow}')
        for c in self.content:
            opts = f'content:"{c}"'
            if self.nocase:
                opts += '; nocase'
            if self.http_uri:
                opts += '; http_uri'
            options.append(opts)
        for p in self.pcre:
            p_escaped = p.replace('"', '\\"')
            options.append(f'pcre:"{p_escaped}"')
        if self.threshold:
            t_parts = [f'type {self.threshold.get("type", "threshold")}']
            if 'track' in self.threshold:
                t_parts.append(f'track {self.threshold["track"]}')
            if 'count' in self.threshold:
                t_parts.append(f'count {self.threshold["count"]}')
            if 'seconds' in self.threshold:
                t_parts.append(f'seconds {self.threshold["seconds"]}')
            options.append(f'threshold:{",".join(t_parts)}')
        options.append(f'classtype:{self.classtype}')
        options.append(f'sid:{self.sid}')
        options.append(f'rev:{self.rev}')
        if self.reference:
            options.append(f'reference:{self.reference}')
        for m in self.metadata:
            options.append(f'metadata:{m}')

        return f'{parts[0]} ({"; ".join(options)}; )'

    def generate_sid(self, seed: str = "") -> int:
        """Generate a deterministic SID from a string seed."""
        if not seed:
            seed = self.msg
        hash_bytes = hashlib.md5(seed.encode()).digest()
        self.sid = int.from_bytes(hash_bytes[:4], 'big') % 900000 + 100000
        return self.sid


def parse_suricata_rule(rule_str: str) -> Optional[SuricataRule]:
    """Parse a Suricata rule string into a SuricataRule object.

    Args:
        rule_str: Raw Suricata rule string.

    Returns:
        SuricataRule object, or None if parsing fails.
    """
    rule_str = rule_str.strip()
    if not rule_str:
        return None

    rule = SuricataRule(raw=rule_str)

    # Parse header: action proto src_ip src_port -> dst_ip dst_port
    header_match = re.match(
        r'(\w+)\s+(\w+)\s+(\S+)\s+(\S+)\s+(->|<>)\s+(\S+)\s+(\S+)\s*\((.*)\)\s*$',
        rule_str, re.DOTALL
    )
    if not header_match:
        return None

    rule.action = header_match.group(1)
    rule.protocol = header_match.group(2)
    rule.src_ip = header_match.group(3)
    rule.src_port = header_match.group(4)
    rule.direction = header_match.group(5)
    rule.dst_ip = header_match.group(6)
    rule.dst_port = header_match.group(7)
    options_str = header_match.group(8)

    # Parse options
    opt_pattern = re.compile(r'(\w+)\s*:\s*(.+?)\s*;(?=\s*(?:\w+\s*:|$))', re.DOTALL)
    for match in opt_pattern.finditer(options_str + ";"):
        key = match.group(1)
        value = match.group(2).strip().strip('"')

        if key == 'msg':
            rule.msg = value
        elif key == 'flow':
            rule.flow = value
        elif key == 'content':
            rule.content.append(value)
        elif key == 'pcre':
            rule.pcre.append(value)
        elif key == 'nocase':
            rule.nocase = True
        elif key == 'http_uri':
            rule.http_uri = True
        elif key == 'classtype':
            rule.classtype = value
        elif key == 'sid':
            try:
                rule.sid = int(value)
            except ValueError:
                pass
        elif key == 'rev':
            try:
                rule.rev = int(value)
            except ValueError:
                pass
        elif key == 'reference':
            rule.reference = value
        elif key == 'metadata':
            rule.metadata.append(value)
        elif key == 'threshold':
            t_dict = {}
            for part in value.split(','):
                if ' ' in part:
                    k, v = part.split(' ', 1)
                    t_dict[k.strip()] = v.strip()
                else:
                    t_dict[part.strip()] = True
            rule.threshold = t_dict

    return rule


def validate_rule_syntax(rule: SuricataRule) -> List[str]:
    """Validate rule syntax without invoking Suricata.

    Returns list of error messages (empty = valid).
    """
    errors = []
    if not rule.action:
        errors.append("Missing action")
    if not rule.protocol:
        errors.append("Missing protocol")
    if not rule.msg:
        errors.append("Missing msg field")
    if rule.sid < 100000:
        errors.append(f"SID {rule.sid} too low (must be >= 100000)")
    if not rule.content and not rule.pcre:
        errors.append("Rule has no content or pcre condition — will match everything")
    return errors
