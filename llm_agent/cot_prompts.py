"""5-Step Structured Chain-of-Thought prompt templates (Section III-E, Algorithm 1).

Each step corresponds to a specific reasoning stage in the agent's generation
process. The templates are designed to be composed sequentially, with each
step's output informing the next.
"""

# ── Step 1: Attack Identification ──────────────────────────────────
STEP1_ATTACK_IDENTIFICATION = """## Step 1: Attack Identification

Analyze the network traffic features above and identify:
1. What type of attack is being conducted?
2. What is the target service or protocol?
3. What are the key behavioral indicators (flow patterns, payload signatures)?
4. What is the adversary's likely goal (data exfiltration, service disruption, unauthorized access)?

Provide a concise characterization of the attack."""

# ── Step 2: Signature Extraction ───────────────────────────────────
STEP2_SIGNATURE_EXTRACTION = """## Step 2: Signature Extraction

Based on the attack identified, extract distinctive signatures:

### Content Signatures
- Identify unique byte sequences from the payload (keywords, URIs, specific strings)
- Note any encoding patterns (hex, base64, URL encoding)
- Extract protocol-specific fields (HTTP method, URI path, DNS query name, etc.)

### Flow-Level Conditions
- Which protocol and port are involved?
- What TCP flags characterize this traffic?
- Are there volumetric thresholds that distinguish this attack?
- What flow direction indicates the attack (to_server? to_client?)?

### PCRE Patterns
- Compose regular expression patterns for variable-length matches
- Account for case insensitivity where appropriate
- Include URL encoding and whitespace variants

List the signatures in structured format."""

# ── Step 3: Rule Drafting ──────────────────────────────────────────
STEP3_RULE_DRAFTING = """## Step 3: Rule Drafting

Draft a complete, syntactically valid Suricata rule using the extracted signatures.

### Guidelines:
1. Use appropriate action (alert) and protocol (http, tcp, dns, etc.)
2. Use $EXTERNAL_NET for source IP, $HOME_NET for destination IP (or vice versa based on direction)
3. Use 'any' for ports unless a specific port is essential
4. Include at least one 'content' keyword for byte-level matching
5. Add 'flow' direction constraints (to_server or to_client)
6. Use 'nocase' for case-insensitive content matching
7. Include a descriptive 'msg' field
8. Assign a unique 'sid' (100001+)
9. Use appropriate 'classtype' (e.g., web-application-attack, attempted-dos, trojan-activity)
10. Reference exemplar rules for structural patterns, but adapt detection logic to this specific attack

Draft the rule below."""

# ── Step 4: Self-Critique ──────────────────────────────────────────
STEP4_SELF_CRITIQUE = """## Step 4: Self-Critique

Critically review the drafted rule:

### Syntax Check
- Are all keywords spelled correctly (msg, content, pcre, flow, classtype, sid, rev)?
- Are all content strings properly quoted?
- Are all semicolons in the correct positions?
- Are parentheses balanced?

### Semantic Check — Will this rule detect the attack?
- Does each content match correspond to an actual byte sequence in the attack traffic?
- Are the pcre patterns correctly formed and not overly broad?
- Do flow conditions correctly constrain the rule to the observed traffic pattern?

### False Positive Check — Will this rule match benign traffic?
- Could legitimate application traffic trigger these content matches?
- Are the IP ranges too broad? (prefer $EXTERNAL_NET / $HOME_NET)
- Should a threshold be added to reduce false positives?
- Are the content keywords specific enough, or could they appear in normal traffic?

### Improvement Suggestions
List specific, actionable improvements to address any issues identified."""

# ── Step 5: Generalization ─────────────────────────────────────────
STEP5_GENERALIZATION = """## Step 5: Generalization

Transform the specific rule into a variant-resistant generalized rule:

### Abstraction
- Replace concrete IP addresses with $EXTERNAL_NET / $HOME_NET
- Replace specific ports with 'any' where the attack is not port-specific
- Remove fixed byte offsets (offset, depth, distance) that assume specific payload positions

### Pattern Generalization
- Broaden content matches to capture encoding variants
  - Example: "UNION SELECT" → pcre:"/UNION\\s+(ALL\\s+)?SELECT/i"
  - Example: "' OR '1'='1" → pcre:"/['\\\"]OR\\s+\\d+\\s*=\\s*\\d+/i"
- Add URL encoding variants for web attacks
- Add whitespace insensitivity where appropriate

### False Positive Mitigation
- Add 'threshold' with count/seconds to suppress isolated matches
- Add additional content matches to increase specificity
- Use 'http_uri' or 'http_header' to constrain the search scope
- Consider 'flowbits' for multi-stage attack correlation

### SQL Injection Pattern Expansion
- UNION SELECT variants: UNION\\s+(ALL\\s+)?SELECT
- Comment obfuscation: /**/, --, #
- String concatenation: ||, CONCAT()
- Boolean-based: OR 1=1, AND 1=1, '1'='1
- Time-based: WAITFOR DELAY, SLEEP(), BENCHMARK()

Produce the generalized rule below."""

# ── Combined CoT Template ──────────────────────────────────────────
FULL_COT_TEMPLATE = """{system_context}

{flow_features}

{payload_features}

{exemplar_rules}

[TASK]
Generate a Suricata rule that detects the attack observed in the traffic features above.

Follow this 5-step reasoning process. Output BOTH a specific rule (tightly matching the observed instance) and a generalized rule (variant-resistant).

{step1}

{step2}

{step3}

{step4}

{step5}

---

## Final Output

<specific_rule>
alert <protocol> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; flow:...; content:"..."; ...; classtype:...; sid:...; rev:1;)
</specific_rule>

<generalized_rule>
alert <protocol> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; flow:...; content:"..."; pcre:"..."; threshold:...; classtype:...; sid:...; rev:1;)
</generalized_rule>"""


def build_cot_prompt(
    system_context: str,
    flow_features: str,
    payload_features: str,
    exemplar_rules: str,
    include_steps: bool = True,
) -> str:
    """Build the full 5-step CoT prompt.

    Args:
        system_context: System prompt for the LLM.
        flow_features: Formatted flow feature text.
        payload_features: Formatted payload text.
        exemplar_rules: Formatted exemplar rules text.
        include_steps: Whether to include the step-by-step reasoning template.

    Returns:
        Complete prompt string.
    """
    if include_steps:
        return FULL_COT_TEMPLATE.format(
            system_context=system_context,
            flow_features=flow_features,
            payload_features=payload_features,
            exemplar_rules=exemplar_rules,
            step1=STEP1_ATTACK_IDENTIFICATION,
            step2=STEP2_SIGNATURE_EXTRACTION,
            step3=STEP3_RULE_DRAFTING,
            step4=STEP4_SELF_CRITIQUE,
            step5=STEP5_GENERALIZATION,
        )

    # Flattened version (for ablation: "flat prompt" condition)
    return f"""{system_context}

{flow_features}

{payload_features}

{exemplar_rules}

[TASK]
Generate a Suricata rule that detects the attack. Output both a specific and a generalized rule:

<specific_rule>
...
</specific_rule>

<generalized_rule>
...
</generalized_rule>"""
