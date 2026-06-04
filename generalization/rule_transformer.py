"""Rule transformer — specific → generalized rule conversion.

Section III-E Step 5 (Generalization): Transforms instance-specific rules
into variant-resistant generalized rules by:
1. Abstracting IPs and ports
2. Expanding PCRE patterns for attack variants
3. Adding false positive mitigations (thresholds, scope restrictions)
4. Removing instance-specific payload offsets and byte positions

This produces the "generalized rule" output described in Algorithm 1.
"""

from typing import Optional, List

from ..utils.suricata_utils import SuricataRule, parse_suricata_rule
from .ip_port_abstractor import abstract_rule
from .pcre_generalizer import generate_pcre_variants, simplify_pcre
from .sqli_patterns import expand_sqli_patterns, add_false_positive_mitigations


class RuleTransformer:
    """Transform specific Suricata rules into generalized variants.

    Args:
        abstract_ip: Replace concrete IPs with $HOME_NET/$EXTERNAL_NET.
        abstract_port: Replace specific ports with 'any' or service variables.
        expand_pcre: Generate PCRE variants for known attack types.
        add_threshold: Add threshold conditions for FPR reduction.
        preserve_flow: Keep flow direction constraints.
    """

    def __init__(
        self,
        abstract_ip: bool = True,
        abstract_port: bool = True,
        expand_pcre: bool = True,
        add_threshold: bool = True,
        preserve_flow: bool = True,
    ):
        self.abstract_ip = abstract_ip
        self.abstract_port = abstract_port
        self.expand_pcre = expand_pcre
        self.add_threshold = add_threshold
        self.preserve_flow = preserve_flow

    def transform(self, rule: SuricataRule, payload_text: str = "") -> SuricataRule:
        """Transform a specific rule into a generalized variant.

        Args:
            rule: The instance-specific SuricataRule.
            payload_text: Original payload text (used for pattern expansion).

        Returns:
            Generalized SuricataRule.
        """
        if rule is None:
            return None

        # Create a copy for transformation
        generalized = SuricataRule(
            action=rule.action,
            protocol=rule.protocol,
            src_ip=rule.src_ip,
            src_port=rule.src_port,
            direction=rule.direction,
            dst_ip=rule.dst_ip,
            dst_port=rule.dst_port,
            msg=rule.msg + " - Generic Pattern",
            flow=rule.flow,
            content=list(rule.content),
            pcre=list(rule.pcre),
            nocase=rule.nocase,
            http_uri=rule.http_uri,
            classtype=rule.classtype,
            sid=rule.sid + 1,  # new SID for generalized rule
            rev=1,
        )

        # Step 1: Abstract IPs and ports
        if self.abstract_ip or self.abstract_port:
            generalized = abstract_rule(generalized)

        # Step 2: Generate PCRE variants
        if self.expand_pcre:
            generalized = generate_pcre_variants(generalized)

            # Also generate from payload text
            if payload_text:
                sqli_patterns = expand_sqli_patterns(payload_text)
                for p in sqli_patterns:
                    if p not in generalized.pcre:
                        generalized.pcre.append(p)

        # Step 3: Deduplicate and simplify PCRE
        generalized.pcre = simplify_pcre(generalized.pcre)

        # Step 4: Add FPR mitigations
        generalized.pcre = add_false_positive_mitigations(generalized.pcre)

        # Step 5: Add threshold for FPR reduction
        if self.add_threshold:
            # Only add threshold if rule is broad (multiple content/pcre)
            if len(generalized.content) >= 2 or len(generalized.pcre) >= 2:
                generalized.threshold = {
                    'type': 'threshold',
                    'track': 'by_src',
                    'count': 3,
                    'seconds': 30,
                }

        # Step 6: Ensure http_uri scope for web rules
        if generalized.protocol == 'http' and not generalized.http_uri:
            # Check if content patterns are URI-related
            uri_indicators = ['/', '?', '=', 'php', 'asp', 'jsp', 'cmd', 'exec']
            for content in generalized.content:
                if any(ind in content.lower() for ind in uri_indicators):
                    generalized.http_uri = True
                    break

        # Generate deterministic SID
        generalized.generate_sid(f"generalized_{rule.sid}_{generalized.msg}")

        return generalized

    def transform_rule_string(
        self,
        rule_str: str,
        payload_text: str = "",
    ) -> Optional[str]:
        """Transform a rule string, returning the generalized rule string.

        Args:
            rule_str: Raw Suricata rule string.
            payload_text: Original payload text.

        Returns:
            Generalized rule string, or None if parsing fails.
        """
        rule = parse_suricata_rule(rule_str)
        if rule is None:
            return None
        generalized = self.transform(rule, payload_text)
        return generalized.to_string() if generalized else None
