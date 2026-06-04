"""Cross-modal structured prompt builder.

Constructs the formatted prompt that combines flow features, payload text,
retrieved exemplar rules, and the 5-step CoT task template into the final
LLM input (Section III-C.3).
"""

from typing import List, Optional, Dict, Any
import numpy as np

from ..utils.suricata_utils import SuricataRule


class PromptBuilder:
    """Build structured prompts from dual-channel features and exemplars.

    Args:
        max_tokens: Maximum token count for the full prompt (approximate).
        include_flow: Whether to include flow statistical features.
        include_payload: Whether to include payload text.
    """

    SYSTEM_TEMPLATE = """[SYSTEM]
You are a Suricata rule generation agent specialized in network intrusion detection.
Your task is to analyze network traffic features extracted from PCAP data and generate
high-quality Suricata rules that detect the observed attack patterns.

Critical requirements:
1. Rules MUST be syntactically valid Suricata format
2. Use $EXTERNAL_NET and $HOME_NET variables for IP addresses
3. Use 'any' for ports unless specific ports are essential for detection
4. Include at least one 'content' or 'pcre' keyword for signature matching
5. Add appropriate flow direction constraints (to_server / to_client)
6. BALANCE detection coverage against false positive risk — avoid overly broad patterns
7. Follow the structured 5-step reasoning process below"""

    CTI_TEMPLATE = """
[ANALYST-VIEW STRUCTURED CTI]
{cti_text}"""

    FLOW_TEMPLATE = """
[FLOW FEATURES]
duration: {duration:.2f}s | pkts_fwd: {pkts_fwd} | pkts_bwd: {pkts_bwd}
bytes_fwd: {bytes_fwd} | bytes_bwd: {bytes_bwd}
iat_mean_fwd: {iat_mean:.3f}s | iat_std_fwd: {iat_std:.3f}s
tcp_flags: SYN={syn:.2f}, ACK={ack:.2f}, PSH={psh:.2f}, FIN={fin:.2f}, RST={rst:.2f}
protocol: {transport_proto} | app_protocol: {app_protocol}"""

    PAYLOAD_TEMPLATE = """
[PAYLOAD]
{payload_text}"""

    EXEMPLAR_TEMPLATE = """
[RETRIEVED EXEMPLARS]
{exemplar_rules}"""

    TASK_TEMPLATE = """
[TASK]
Generate a Suricata rule that detects the attack observed in the traffic features above.
Follow this 5-step reasoning process:

Step 1 — Attack Identification:
  - Identify the attack type based on flow patterns and payload content
  - Describe key characteristics: attack vector, target service, behavioral indicators

Step 2 — Signature Extraction:
  - Extract distinctive content patterns from the payload (keywords, URI structures, hex patterns)
  - Identify flow-level conditions (protocol, port, direction, flag patterns)
  - Compose PCRE (regular expression) patterns for complex matching

Step 3 — Rule Drafting:
  - Draft a complete Suricata rule with proper header and options
  - Use exemplar rules as structural references (adapt detection logic, do not copy)
  - Ensure all content matches are necessary and sufficient

Step 4 — Self-Critique:
  - Review the rule for syntax errors (missing semicolons, invalid keywords, unclosed quotes)
  - Check: does the rule match ONLY the attack traffic? Would it trigger on benign traffic?
  - Check: are IP addresses and ports properly abstracted ($EXTERNAL_NET, $HOME_NET, any)?

Step 5 — Generalization:
  - Replace instance-specific patterns (concrete IPs, exact byte offsets) with variables
  - Broaden PCRE patterns to capture attack variants (e.g., URL encoding variants, case variations)
  - Add threshold conditions to suppress false positives from legitimate traffic

Output format:
<specific_rule>
alert <proto> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; ...; sid:...; rev:1;)
</specific_rule>

<generalized_rule>
alert <proto> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; ...; sid:...; rev:1;)
</generalized_rule>"""

    def __init__(
        self,
        max_tokens: int = 2048,
        include_flow: bool = True,
        include_payload: bool = True,
        include_cti: bool = True,
    ):
        self.max_tokens = max_tokens
        self.include_flow = include_flow
        self.include_payload = include_payload
        self.include_cti = include_cti

    def build(
        self,
        flow_features: Optional[np.ndarray],
        payload_text: str,
        exemplars: Optional[List[SuricataRule]] = None,
        cti: Any = None,
        include_task: bool = True,
    ) -> str:
        """Build the full structured prompt.

        Args:
            flow_features: (4, 38) aggregated flow statistics, or None.
            payload_text: Extracted payload content text.
            exemplars: List of retrieved SuricataRule exemplars.
            cti: StructuredCTI object (Analyst-View CTI) or None.
            include_task: Whether to append the 5-step task template (set False for training).

        Returns:
            Complete prompt string ready for LLM input.
        """
        parts = [self.SYSTEM_TEMPLATE.strip()]

        # Structured CTI (primary input for the LLM agent)
        if self.include_cti and cti is not None:
            cti_str = self._format_cti(cti)
            if cti_str:
                parts.append(cti_str)

        # Flow features
        if self.include_flow and flow_features is not None:
            flow_str = self._format_flow_features(flow_features)
            if flow_str:
                parts.append(flow_str)

        # Payload
        if self.include_payload and payload_text:
            parts.append(
                self.PAYLOAD_TEMPLATE.format(payload_text=payload_text).strip()
            )

        # Exemplars
        if exemplars:
            exemplar_str = self._format_exemplars(exemplars)
            parts.append(
                self.EXEMPLAR_TEMPLATE.format(exemplar_rules=exemplar_str).strip()
            )

        # Task
        if include_task:
            parts.append(self.TASK_TEMPLATE.strip())

        prompt = "\n\n".join(parts)
        # Truncate to approximate token limit (4 chars/token is rough estimate)
        if len(prompt) > self.max_tokens * 4:
            prompt = prompt[:self.max_tokens * 4]
        return prompt

    def _format_cti(self, cti: Any) -> str:
        """Format the Structured CTI into the prompt template.

        Args:
            cti: StructuredCTI object or dict with CTI data.

        Returns:
            Formatted CTI text for the prompt.
        """
        try:
            # Use compact text representation if available
            if hasattr(cti, 'to_compact_text'):
                cti_text = cti.to_compact_text()
            elif hasattr(cti, 'to_dict'):
                import json
                cti_text = json.dumps(cti.to_dict(), indent=2, ensure_ascii=False)
            elif isinstance(cti, dict):
                import json
                cti_text = json.dumps(cti, indent=2, ensure_ascii=False)
            else:
                cti_text = str(cti)

            return self.CTI_TEMPLATE.format(cti_text=cti_text).strip()
        except Exception:
            return ""

    def _format_flow_features(self, features: np.ndarray) -> str:
        """Format the aggregated flow feature matrix into the prompt template.

        Args:
            features: (4, 38) array: rows = [mean, std, max, min].

        Returns:
            Formatted string of flow statistics.
        """
        if features.size == 0:
            return ""

        mean = features[0]  # mean row
        # Use mean row for the template; key indices:
        # indices (based on flow_extractor.py ordering):
        #   0: duration_ms, 7: src2dst_first_seen, 8: iat_mean, 9: iat_std
        #   10: src2dst_packets, 11: dst2src_packets
        #   12: src2dst_bytes, 13: dst2src_bytes
        #   22: syn_pkts, 26: ack_pkts, 20: psh_pkts, 24: fin_pkts, 25: rst_pkts
        #   36: transport_proto, app_protocol from mean

        try:
            return self.FLOW_TEMPLATE.format(
                duration=float(mean[0]) / 1000.0 if mean[0] > 0 else 0.0,  # ms -> s
                pkts_fwd=int(mean[10]) if len(mean) > 10 else 0,
                pkts_bwd=int(mean[11]) if len(mean) > 11 else 0,
                bytes_fwd=int(mean[12]) if len(mean) > 12 else 0,
                bytes_bwd=int(mean[13]) if len(mean) > 13 else 0,
                iat_mean=float(mean[8]) / 1000.0 if len(mean) > 8 else 0.0,
                iat_std=float(mean[9]) / 1000.0 if len(mean) > 9 else 0.0,
                syn=float(mean[22]) if len(mean) > 22 else 0.0,
                ack=float(mean[26]) if len(mean) > 26 else 0.0,
                psh=float(mean[20]) if len(mean) > 20 else 0.0,
                fin=float(mean[24]) if len(mean) > 24 else 0.0,
                rst=float(mean[25]) if len(mean) > 25 else 0.0,
                transport_proto='TCP' if (mean[37] if len(mean) > 37 else 6) == 6 else 'UDP',
                app_protocol='HTTP' if (mean[31] if len(mean) > 31 else 0) else 'Unknown',
            ).strip()
        except (IndexError, ValueError):
            return "[Flow features unavailable]"

    def _format_exemplars(self, exemplars: List[SuricataRule]) -> str:
        """Format retrieved exemplar rules for the prompt."""
        lines = []
        for i, rule in enumerate(exemplars, 1):
            lines.append(f"Rule {i}: {rule.to_string()}")
        return "\n".join(lines)

    def build_training_prompt(
        self,
        flow_features: Optional[np.ndarray],
        payload_text: str,
        exemplars: Optional[List[SuricataRule]] = None,
        cti: Any = None,
    ) -> str:
        """Build a prompt for training (excludes the task template for teacher forcing).

        Returns:
            Prompt string ending right before the model should generate the rule.
        """
        prompt = self.build(flow_features, payload_text, exemplars, cti=cti, include_task=False)
        # Append the task instruction without the output format spec
        prompt += "\n\n[TASK]\nGenerate a Suricata rule that detects the attack observed above.\n\n"
        prompt += "<specific_rule>\n"
        return prompt
