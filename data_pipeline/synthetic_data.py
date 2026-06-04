"""Synthetic training data generation — Dataset D1 (Section IV-A).

Uses GPT-4o in reverse-generation mode: given a Suricata rule,
generate a plausible (flow_features, payload_text) pair.
This creates 2000 synthetic training samples from the 2000
community rules in the knowledge base.

D1 serves two purposes:
  1. Augmenting limited real traffic data for LoRA fine-tuning
  2. Teaching the model the mapping from traffic patterns to rules
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import random
import numpy as np

from ..utils.suricata_utils import SuricataRule
from ..utils.logging import get_logger


REVERSE_GENERATION_PROMPT = """You are a network traffic analyst. Given a Suricata IDS rule, generate a realistic HTTP request payload and flow feature description that would trigger this rule.

Rule: {rule_text}

Attack Type: {attack_type}

Generate:
1. HTTP REQUEST PAYLOAD (realistic, include the malicious patterns from the rule):
```
<http_request>
```

2. FLOW FEATURES (describe the flow characteristics in text format):
   - Duration (seconds)
   - Total packets sent/received
   - Total bytes sent/received
   - Packet rate (pps)
   - Bytes per second
   - Protocol flags distribution
   - TCP window sizes

Output format:
```json
{{
    "http_request": "...",
    "flow_description": "...",
    "attack_type": "{attack_type}"
}}
```
"""


@dataclass
class SyntheticSample:
    """A synthetic training sample (D1)."""
    sample_id: str
    attack_type: str
    rule_text: str
    http_payload: str
    flow_description: str
    metadata: dict


def build_reverse_prompt(rule: SuricataRule) -> str:
    """Build the reverse-generation prompt for one rule.

    Args:
        rule: A Suricata rule from the knowledge base.

    Returns:
        Formatted prompt string for GPT-4o.
    """
    attack_type = rule.metadata.get('attack_type', 'Unknown') if hasattr(rule, 'metadata') else 'Unknown'
    rule_text = rule.to_string()

    return REVERSE_GENERATION_PROMPT.format(
        rule_text=rule_text,
        attack_type=attack_type,
    )


def parse_synthetic_response(
    response: str,
    rule: SuricataRule,
    sample_id: str,
) -> Optional[SyntheticSample]:
    """Parse GPT-4o response into a SyntheticSample.

    Args:
        response: Raw text response from GPT-4o.
        rule: The source Suricata rule.
        sample_id: Unique identifier for this sample.

    Returns:
        SyntheticSample or None if parsing fails.
    """
    try:
        # Extract JSON block
        json_start = response.find('```json')
        if json_start == -1:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
        else:
            json_start = response.find('{', json_start)
            json_end = response.rfind('```')
            if json_end == -1:
                json_end = len(response)

        json_str = response[json_start:json_end].strip()
        data = json.loads(json_str)

        return SyntheticSample(
            sample_id=sample_id,
            attack_type=data.get('attack_type', 'Unknown'),
            rule_text=rule.to_string(),
            http_payload=data.get('http_request', ''),
            flow_description=data.get('flow_description', ''),
            metadata={
                'source_rule_sid': rule.sid,
                'source_rule_msg': rule.msg,
            },
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None


def quality_check_sample(
    sample: SyntheticSample,
    rule: SuricataRule,
) -> bool:
    """Verify that the synthetic sample is consistent with the source rule.

    Checks:
      1. HTTP payload contains at least one content match from the rule
      2. Payload length is between 100 and 4096 characters
      3. Attack type is present and valid

    Args:
        sample: Generated synthetic sample.
        rule: Source Suricata rule.

    Returns:
        True if the sample passes quality checks.
    """
    # Check payload length
    if len(sample.http_payload) < 100 or len(sample.http_payload) > 4096:
        return False

    # Check for at least one content match
    content_matches = [c for c in rule.content if c and c != '""']
    if content_matches:
        has_match = any(
            c.lower().strip('"') in sample.http_payload.lower()
            for c in content_matches
        )
        if not has_match:
            return False

    # Check attack type
    valid_types = {
        'Benign', 'Brute Force', 'SQL Injection', 'XSS', 'DDoS',
        'Infiltration', 'Botnet', 'Port Scan', 'Web Attack',
    }
    if sample.attack_type not in valid_types:
        return False

    return True


def generate_synthetic_dataset(
    rules: List[SuricataRule],
    llm_call_fn,
    n_samples: int = 2000,
    quality_filter: bool = True,
    seed: int = 42,
) -> List[SyntheticSample]:
    """Generate D1 synthetic dataset from community rules.

    Args:
        rules: List of Suricata rules from the knowledge base.
        llm_call_fn: Async function(rule_text, prompt) -> generated text.
                     Use GPT-4o or another capable LLM for high quality.
        n_samples: Target number of samples (default 2000).
        quality_filter: Whether to filter by consistency check.
        seed: Random seed for rule selection.

    Returns:
        List of SyntheticSample objects.
    """
    logger = get_logger("pcap2rule.synthetic")
    rng = random.Random(seed)

    if len(rules) < n_samples:
        logger.warning(
            f"Only {len(rules)} rules available for {n_samples} targets; "
            f"will sample with replacement"
        )
        selected = rng.choices(rules, k=n_samples)
    else:
        selected = rng.sample(rules, n_samples)

    samples = []
    failed = 0

    for idx, rule in enumerate(selected):
        prompt = build_reverse_prompt(rule)
        sample_id = f"D1_{idx:04d}"

        try:
            response = llm_call_fn(prompt)
            sample = parse_synthetic_response(response, rule, sample_id)

            if sample is None:
                failed += 1
                continue

            if quality_filter and not quality_check_sample(sample, rule):
                failed += 1
                continue

            samples.append(sample)

            if (idx + 1) % 200 == 0:
                logger.info(
                    f"Generated {len(samples)}/{idx + 1} samples "
                    f"({failed} failed QC)"
                )

        except Exception as e:
            logger.warning(f"Failed to generate sample {sample_id}: {e}")
            failed += 1

    logger.info(
        f"D1 generation complete: {len(samples)} valid, {failed} failed "
        f"({len(samples) / max(1, len(selected)) * 100:.1f}% success rate)"
    )
    return samples


def save_synthetic_dataset(
    samples: List[SyntheticSample],
    output_path: str,
):
    """Save synthetic dataset to JSON file.

    Args:
        samples: List of SyntheticSample objects.
        output_path: Path to output JSON file.
    """
    data = []
    for s in samples:
        data.append({
            'sample_id': s.sample_id,
            'attack_type': s.attack_type,
            'rule_text': s.rule_text,
            'http_payload': s.http_payload,
            'flow_description': s.flow_description,
            'metadata': s.metadata,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_synthetic_dataset(path: str) -> List[SyntheticSample]:
    """Load synthetic dataset from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of SyntheticSample objects.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(SyntheticSample(
            sample_id=item['sample_id'],
            attack_type=item['attack_type'],
            rule_text=item['rule_text'],
            http_payload=item['http_payload'],
            flow_description=item['flow_description'],
            metadata=item.get('metadata', {}),
        ))
    return samples
