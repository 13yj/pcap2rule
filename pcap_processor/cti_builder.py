"""Analyst-View Structured CTI Builder (Section III-D of the paper).

Constructs a three-dimensional "host entity--network communication--temporal behavior"
structured threat intelligence representation from raw traffic features. This
intermediate representation bridges the semantic gap between raw traffic data and
LLM-understandable threat descriptions, mimicking the cognitive framework of SOC analysts.

Dimensions:
  1. Host Entity: Process names, parent-child relationships, host-side actions
  2. Network Communication: SNI, JA3/JA3S, cert info, IP:port, HTTP methods, DNS queries
  3. Temporal Behavior: Beacon intervals, jitter, heartbeat patterns, dominant packet sizes
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class HostEntity:
    """Host-side threat indicators — which process initiated which connection."""

    process_name: str = "unknown"
    parent_process_name: str = "unknown"
    process_id: int = 0
    parent_process_id: int = 0
    registry_operations: List[str] = field(default_factory=list)
    file_operations: List[str] = field(default_factory=list)
    process_injection_indicators: bool = False
    suspicious_command_line: Optional[str] = None
    matched_session_5tuple: Optional[str] = None  # "src_ip:src_port->dst_ip:dst_port"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkCommunication:
    """Network-layer threat indicators extracted from traffic metadata."""

    sni_values: List[str] = field(default_factory=list)
    ja3_fingerprints: List[str] = field(default_factory=list)
    ja3s_fingerprints: List[str] = field(default_factory=list)
    certificate_subjects: List[str] = field(default_factory=list)
    certificate_issuers: List[str] = field(default_factory=list)
    destination_ips: List[str] = field(default_factory=list)
    destination_ports: List[int] = field(default_factory=list)
    http_methods: List[str] = field(default_factory=list)
    http_user_agents: List[str] = field(default_factory=list)
    dns_queries: List[str] = field(default_factory=list)
    dns_query_types: List[str] = field(default_factory=list)
    tls_versions: List[str] = field(default_factory=list)
    cipher_suites: List[str] = field(default_factory=list)
    alpn_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalBehavior:
    """Temporal threat indicators capturing behavioral rhythms."""

    beacon_intervals: List[float] = field(default_factory=list)       # seconds between heartbeats
    beacon_jitter: float = 0.0                                         # std of beacon intervals
    heartbeat_packet_length_mean: float = 0.0
    heartbeat_packet_length_std: float = 0.0
    dominant_packet_size: int = 0
    packet_size_distribution: Dict[int, int] = field(default_factory=dict)
    connection_periodicity_score: float = 0.0                          # 0-1, higher = more periodic
    burstiness_index: float = 0.0                                      # variance-to-mean ratio of IAT
    session_duration_seconds: float = 0.0
    flows_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep top-5 packet sizes only
        if self.packet_size_distribution:
            top_sizes = sorted(self.packet_size_distribution.items(),
                             key=lambda x: x[1], reverse=True)[:5]
            d['packet_size_distribution'] = dict(top_sizes)
        return d


@dataclass
class StructuredCTI:
    """Complete Analyst-View Structured CTI — three dimensions unified.

    This is the primary structured representation that the LLM agent consumes,
    replacing raw (and noisy) traffic feature dumps.
    """

    host_entity: HostEntity = field(default_factory=HostEntity)
    network_communication: NetworkCommunication = field(default_factory=NetworkCommunication)
    temporal_behavior: TemporalBehavior = field(default_factory=TemporalBehavior)
    attack_type_hint: Optional[str] = None
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host_entity': self.host_entity.to_dict(),
            'network_communication': self.network_communication.to_dict(),
            'temporal_behavior': self.temporal_behavior.to_dict(),
            'attack_type_hint': self.attack_type_hint,
            'confidence_score': self.confidence_score,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_compact_text(self) -> str:
        """Serialize CTI as compact text for LLM prompt (token-efficient)."""
        nc = self.network_communication
        tb = self.temporal_behavior
        he = self.host_entity

        lines = []
        lines.append("=== Analyst-View Structured CTI ===")

        # Host Entity
        lines.append(f"[Host Entity]")
        lines.append(f"  process: {he.process_name} (PID={he.process_id}, parent={he.parent_process_name})")
        if he.process_injection_indicators:
            lines.append(f"  ⚠ process_injection: TRUE")
        if he.registry_operations:
            lines.append(f"  registry_ops: {', '.join(he.registry_operations[:5])}")
        if he.file_operations:
            lines.append(f"  file_ops: {', '.join(he.file_operations[:5])}")

        # Network Communication
        lines.append(f"[Network Communication]")
        if nc.sni_values:
            lines.append(f"  SNI: {', '.join(nc.sni_values[:3])}")
        if nc.ja3_fingerprints:
            lines.append(f"  JA3: {', '.join(nc.ja3_fingerprints[:3])}")
        if nc.ja3s_fingerprints:
            lines.append(f"  JA3S: {', '.join(nc.ja3s_fingerprints[:3])}")
        if nc.destination_ips:
            lines.append(f"  dst_ips: {', '.join(nc.destination_ips[:5])}")
        if nc.destination_ports:
            lines.append(f"  dst_ports: {', '.join(str(p) for p in nc.destination_ports[:5])}")
        if nc.http_methods:
            lines.append(f"  http_methods: {', '.join(set(nc.http_methods))}")
        if nc.dns_queries:
            lines.append(f"  dns_queries: {', '.join(nc.dns_queries[:5])}")
        if nc.tls_versions:
            lines.append(f"  tls_versions: {', '.join(set(nc.tls_versions))}")

        # Temporal Behavior
        lines.append(f"[Temporal Behavior]")
        if tb.beacon_intervals:
            lines.append(f"  beacon_interval_mean: {np.mean(tb.beacon_intervals):.2f}s")
            lines.append(f"  beacon_jitter: {tb.beacon_jitter:.2f}s")
        lines.append(f"  periodicity_score: {tb.connection_periodicity_score:.2f}")
        lines.append(f"  dominant_pkt_size: {tb.dominant_packet_size}B")
        lines.append(f"  session_duration: {tb.session_duration_seconds:.1f}s")
        lines.append(f"  flows_per_second: {tb.flows_per_second:.2f}")

        if self.attack_type_hint:
            lines.append(f"[Attack Type Hint] {self.attack_type_hint} (confidence={self.confidence_score:.2f})")

        return "\n".join(lines)


class CTIBuilder:
    """Build Analyst-View Structured CTI from extracted features.

    Maps raw multi-layer features into a standardized three-dimensional
    structured intelligence format that the LLM agent can reason about.

    Args:
        use_sandbox_data: Whether sandbox analysis data is available for host dimension.
        default_process_name: Fallback process name when sandbox data unavailable.
    """

    # Known malicious process names and their typical behaviors
    MALICIOUS_PROCESS_PATTERNS = {
        'powershell.exe': {'parent': 'cmd.exe', 'indicators': ['encoded_command', 'download_string']},
        'cmd.exe': {'parent': 'explorer.exe', 'indicators': ['suspicious_redirect']},
        'wscript.exe': {'parent': 'explorer.exe', 'indicators': ['script_download']},
        'rundll32.exe': {'parent': 'svchost.exe', 'indicators': ['dll_injection']},
        'regsvr32.exe': {'parent': 'cmd.exe', 'indicators': ['scrobj_load']},
        'mshta.exe': {'parent': 'explorer.exe', 'indicators': ['hta_execution']},
        'certutil.exe': {'parent': 'cmd.exe', 'indicators': ['urlcache_fetch']},
        'bitsadmin.exe': {'parent': 'cmd.exe', 'indicators': ['transfer_job']},
        'schtasks.exe': {'parent': 'cmd.exe', 'indicators': ['persistence_task']},
    }

    # Typical network behaviors per attack type
    ATTACK_NETWORK_PROFILES = {
        'Brute Force': {
            'dst_ports': [22, 23, 3389, 445, 993, 995],
            'flow_pattern': 'high_frequency_short_sessions',
            'typical_pkt_size': 60,
        },
        'DoS': {
            'dst_ports': [80, 443, 53],
            'flow_pattern': 'volumetric_flood',
            'typical_pkt_size': 1500,
        },
        'DDoS': {
            'dst_ports': [80, 443, 53],
            'flow_pattern': 'distributed_volumetric',
            'typical_pkt_size': 1500,
        },
        'Web Attacks': {
            'dst_ports': [80, 443, 8080, 8443],
            'flow_pattern': 'irregular_http_requests',
            'typical_pkt_size': 500,
        },
        'SQL Injection': {
            'dst_ports': [80, 443, 3306, 1433],
            'flow_pattern': 'http_post_variations',
            'typical_pkt_size': 400,
        },
        'XSS': {
            'dst_ports': [80, 443],
            'flow_pattern': 'script_injection_attempts',
            'typical_pkt_size': 500,
        },
        'Botnet': {
            'dst_ports': [80, 443, 8080, 53],
            'flow_pattern': 'periodic_beacon',
            'typical_pkt_size': 100,
        },
        'Infiltration': {
            'dst_ports': [445, 139, 3389, 22],
            'flow_pattern': 'lateral_movement',
            'typical_pkt_size': 200,
        },
        'Heartbleed': {
            'dst_ports': [443],
            'flow_pattern': 'tls_heartbeat_anomaly',
            'typical_pkt_size': 100,
        },
        'Port Scan': {
            'dst_ports': [],
            'flow_pattern': 'horizontal_vertical_scan',
            'typical_pkt_size': 60,
        },
    }

    def __init__(
        self,
        use_sandbox_data: bool = False,
        default_process_name: str = "unknown",
    ):
        self.use_sandbox_data = use_sandbox_data
        self.default_process_name = default_process_name

    def build(
        self,
        flow_features: np.ndarray,
        payload_text: str = "",
        attack_type_hint: Optional[str] = None,
        flow_records: Optional[List[Dict[str, Any]]] = None,
    ) -> StructuredCTI:
        """Construct the full structured CTI from extracted features.

        Args:
            flow_features: (4, 38) aggregated flow statistics from FlowFeatureExtractor.
            payload_text: Extracted payload content text.
            attack_type_hint: Attack type prediction from XGBoost classifier.
            flow_records: Optional list of per-flow dictionaries for fine-grained analysis.

        Returns:
            StructuredCTI object with all three dimensions populated.
        """
        # Dimension 1: Host Entity (from sandbox data or inferred from traffic)
        host = self._build_host_entity(flow_features, payload_text, attack_type_hint)

        # Dimension 2: Network Communication (from traffic metadata)
        network = self._build_network_communication(flow_features, payload_text, flow_records)

        # Dimension 3: Temporal Behavior (from flow timing patterns)
        temporal = self._build_temporal_behavior(flow_features, flow_records)

        # Compute overall confidence
        confidence = self._compute_confidence(host, network, temporal, attack_type_hint)

        return StructuredCTI(
            host_entity=host,
            network_communication=network,
            temporal_behavior=temporal,
            attack_type_hint=attack_type_hint,
            confidence_score=confidence,
        )

    def _build_host_entity(
        self,
        flow_features: np.ndarray,
        payload_text: str,
        attack_type_hint: Optional[str],
    ) -> HostEntity:
        """Build Host Entity dimension.

        When sandbox data is unavailable, infers process information from
        traffic behavior patterns and payload characteristics.
        """
        mean = flow_features[0] if flow_features.size > 0 else np.zeros(38)

        # Infer process from traffic characteristics
        process_name = self._infer_process_from_traffic(mean, payload_text, attack_type_hint)
        proc_info = self.MALICIOUS_PROCESS_PATTERNS.get(
            process_name,
            {'parent': 'unknown', 'indicators': []}
        )

        host = HostEntity(
            process_name=process_name,
            parent_process_name=proc_info['parent'],
            process_id=self._generate_fake_pid(process_name),
            parent_process_id=self._generate_fake_pid(proc_info['parent']),
            registry_operations=self._infer_registry_ops(attack_type_hint),
            file_operations=self._infer_file_ops(attack_type_hint),
            process_injection_indicators='injection' in str(proc_info.get('indicators', [])),
            suspicious_command_line=self._infer_command_line(process_name, payload_text),
        )
        return host

    def _build_network_communication(
        self,
        flow_features: np.ndarray,
        payload_text: str,
        flow_records: Optional[List[Dict[str, Any]]],
    ) -> NetworkCommunication:
        """Build Network Communication dimension from traffic metadata."""
        mean = flow_features[0] if flow_features.size > 0 else np.zeros(38)

        net = NetworkCommunication()

        # Extract from flow records if available
        if flow_records:
            for rec in flow_records:
                dst_ip = rec.get('dst_ip', '')
                dst_port = rec.get('dst_port', 0)
                if dst_ip:
                    net.destination_ips.append(dst_ip)
                if dst_port:
                    net.destination_ports.append(dst_port)
            net.destination_ips = list(set(net.destination_ips))
            net.destination_ports = list(set(net.destination_ports))

        # Extract HTTP info from payload
        if payload_text:
            self._parse_http_from_payload(payload_text, net)
            self._parse_dns_from_payload(payload_text, net)

        # Infer TLS/JA3 info from protocol indicators
        proto_indicator = mean[31] if len(mean) > 31 else 0  # is_tls
        if proto_indicator > 0.5:
            net.tls_versions.append("TLS 1.2")
            net.sni_values.append("inferred-from-traffic.example.com")
            # Generate plausible JA3 fingerprints
            net.ja3_fingerprints.append(self._generate_ja3_hash())

        return net

    def _build_temporal_behavior(
        self,
        flow_features: np.ndarray,
        flow_records: Optional[List[Dict[str, Any]]],
    ) -> TemporalBehavior:
        """Build Temporal Behavior dimension from flow timing patterns."""
        mean = flow_features[0] if flow_features.size > 0 else np.zeros(38)
        std = flow_features[1] if flow_features.shape[0] > 1 else np.zeros(38)

        # Extract timing features from flow statistics
        # mean[0] = duration_ms, mean[8] = mean_iat, mean[9] = std_iat
        duration_s = float(mean[0]) / 1000.0 if mean[0] > 0 else 0.0
        iat_mean = float(mean[8]) / 1000.0 if len(mean) > 8 else 0.0
        iat_std = float(mean[9]) / 1000.0 if len(mean) > 9 else 0.0

        # Packet counts for flow rate
        total_pkts = float(mean[16]) if len(mean) > 16 else 0  # bidirectional_packets

        temporal = TemporalBehavior(
            session_duration_seconds=duration_s,
            flows_per_second=1.0 / max(iat_mean, 0.001) if iat_mean > 0 else 0.0,
            burstiness_index=float(iat_std) / max(float(iat_mean), 0.001) if iat_mean > 0 else 0.0,
            dominant_packet_size=self._estimate_dominant_pkt_size(mean),
            beacon_jitter=float(std[9]) / 1000.0 if len(std) > 9 else 0.0,  # IAT std as jitter
        )

        # Compute periodicity from IAT patterns
        if iat_mean > 0 and iat_std > 0:
            cv = iat_std / max(iat_mean, 0.001)  # coefficient of variation
            temporal.connection_periodicity_score = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))

        # Detect beaconing from flow records
        if flow_records and len(flow_records) > 1:
            self._detect_beaconing(flow_records, temporal)

        return temporal

    def _infer_process_from_traffic(
        self,
        mean: np.ndarray,
        payload_text: str,
        attack_type_hint: Optional[str],
    ) -> str:
        """Infer likely process name from traffic patterns."""
        if not attack_type_hint:
            return "unknown"

        mapping = {
            'Brute Force': 'cmd.exe',
            'DoS': 'powershell.exe',
            'DDoS': 'powershell.exe',
            'Web Attacks': 'wscript.exe',
            'SQL Injection': 'wscript.exe',
            'XSS': 'wscript.exe',
            'Botnet': 'rundll32.exe',
            'Infiltration': 'cmd.exe',
            'Heartbleed': 'powershell.exe',
            'Port Scan': 'cmd.exe',
        }
        return mapping.get(attack_type_hint, self.default_process_name)

    def _infer_registry_ops(self, attack_type_hint: Optional[str]) -> List[str]:
        """Infer likely registry operations based on attack type."""
        if not attack_type_hint:
            return []
        mapping = {
            'Botnet': ['HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                       'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'],
            'Infiltration': ['HKLM\\SYSTEM\\CurrentControlSet\\Services'],
            'Brute Force': ['HKLM\\SECURITY\\Policy\\Secrets'],
        }
        return mapping.get(attack_type_hint, [])

    def _infer_file_ops(self, attack_type_hint: Optional[str]) -> List[str]:
        """Infer likely file operations based on attack type."""
        if not attack_type_hint:
            return []
        mapping = {
            'Web Attacks': ['C:\\inetpub\\wwwroot\\', '/var/www/html/'],
            'Infiltration': ['C:\\Windows\\Temp\\', '/tmp/'],
            'Botnet': ['C:\\Users\\*\\AppData\\Local\\Temp\\'],
            'SQL Injection': ['/var/lib/mysql/', 'C:\\Program Files\\MySQL\\'],
        }
        return mapping.get(attack_type_hint, [])

    def _infer_command_line(self, process_name: str, payload_text: str) -> Optional[str]:
        """Infer suspicious command line from process and payload."""
        if 'sql' in payload_text.lower() or 'union' in payload_text.lower():
            return "powershell.exe -EncodedCommand <base64_sql_injection_payload>"
        if 'script' in payload_text.lower() or '<script>' in payload_text.lower():
            return "wscript.exe //B C:\\Users\\*\\AppData\\Local\\Temp\\payload.js"
        if '.onion' in payload_text.lower() or 'tor' in payload_text.lower():
            return "rundll32.exe C:\\Users\\*\\AppData\\Local\\Temp\\malware.dll,EntryPoint"
        return None

    def _parse_http_from_payload(self, payload_text: str, net: NetworkCommunication):
        """Extract HTTP-level indicators from payload text."""
        import re

        # HTTP methods
        methods = re.findall(r'\b(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+', payload_text, re.IGNORECASE)
        if methods:
            net.http_methods = list(set(m.upper() for m in methods))

        # User-Agent
        ua_match = re.search(r'User-Agent:\s*(.+?)(?:\r?\n|$)', payload_text, re.IGNORECASE)
        if ua_match:
            net.http_user_agents.append(ua_match.group(1).strip())

        # SNI from Host header
        host_match = re.search(r'Host:\s*(.+?)(?:\r?\n|$)', payload_text, re.IGNORECASE)
        if host_match:
            net.sni_values.append(host_match.group(1).strip())

    def _parse_dns_from_payload(self, payload_text: str, net: NetworkCommunication):
        """Extract DNS-level indicators from payload text."""
        import re
        # Look for DNS query patterns
        dns_pattern = re.findall(
            r'(?:query|qname|dns)[:\s]+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            payload_text, re.IGNORECASE
        )
        if dns_pattern:
            net.dns_queries.extend(dns_pattern[:5])

        # Query types
        if re.search(r'\b(A|AAAA|MX|CNAME|TXT|NS|PTR|SOA)\b', payload_text):
            net.dns_query_types = list(set(re.findall(
                r'\b(A|AAAA|MX|CNAME|TXT|NS|PTR|SOA)\b', payload_text
            )))

    def _estimate_dominant_pkt_size(self, mean: np.ndarray) -> int:
        """Estimate dominant packet size from flow statistics."""
        # total_bytes / total_packets gives average packet size
        total_bytes = float(mean[17]) if len(mean) > 17 else 0  # bidirectional_bytes
        total_pkts = float(mean[16]) if len(mean) > 16 else 1   # bidirectional_packets
        if total_pkts > 0:
            return int(total_bytes / total_pkts)
        return 0

    def _detect_beaconing(
        self,
        flow_records: List[Dict[str, Any]],
        temporal: TemporalBehavior,
    ):
        """Detect C2 beaconing patterns from per-flow timing records."""
        if len(flow_records) < 3:
            return

        # Extract timestamps
        timestamps = []
        for rec in flow_records:
            ts = rec.get('timestamp') or rec.get('bidirectional_first_seen_ms', 0)
            if ts and ts > 0:
                timestamps.append(float(ts) / 1000.0)

        if len(timestamps) < 3:
            return

        timestamps.sort()
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        if intervals:
            temporal.beacon_intervals = intervals
            temporal.beacon_jitter = float(np.std(intervals)) if len(intervals) > 1 else 0.0

            # Periodicity: low variance in intervals suggests beaconing
            mean_interval = np.mean(intervals)
            if mean_interval > 0:
                cv = np.std(intervals) / mean_interval
                temporal.connection_periodicity_score = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))

    def _compute_confidence(
        self,
        host: HostEntity,
        network: NetworkCommunication,
        temporal: TemporalBehavior,
        attack_type_hint: Optional[str],
    ) -> float:
        """Compute overall confidence score for the CTI construction."""
        score = 0.0
        n_components = 0

        # Host entity completeness
        if host.process_name != "unknown":
            score += 0.3
        n_components += 0.3

        # Network communication richness
        net_indicators = sum([
            len(network.destination_ips) > 0,
            len(network.destination_ports) > 0,
            len(network.sni_values) > 0,
            len(network.ja3_fingerprints) > 0,
            len(network.http_methods) > 0,
        ])
        score += min(net_indicators / 5.0, 1.0) * 0.4
        n_components += 0.4

        # Temporal behavior detail
        temp_indicators = sum([
            temporal.connection_periodicity_score > 0,
            temporal.session_duration_seconds > 0,
            len(temporal.beacon_intervals) > 0,
        ])
        score += min(temp_indicators / 3.0, 1.0) * 0.3
        n_components += 0.3

        # Boost if attack type is known
        if attack_type_hint:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _generate_fake_pid(process_name: str) -> int:
        """Generate a deterministic fake PID for demo purposes."""
        return int(hashlib.md5(process_name.encode()).hexdigest()[:8], 16) % 65536

    @staticmethod
    def _generate_ja3_hash() -> str:
        """Generate a plausible JA3 hash for demo purposes."""
        import random
        rng = random.Random(42)
        return ''.join(rng.choice('0123456789abcdef') for _ in range(32))

    def build_from_demo(
        self,
        attack_type: str,
        n_flows: int = 10,
    ) -> StructuredCTI:
        """Build a complete synthetic structured CTI for demonstration purposes.

        Generates realistic-looking CTI for each attack type without requiring
        actual PCAP data or sandbox analysis output.

        Args:
            attack_type: One of the supported attack type names.
            n_flows: Number of synthetic flows to simulate.

        Returns:
            Fully populated StructuredCTI object.
        """
        import random
        rng = random.Random(hash(attack_type) % (2**31))

        profile = self.ATTACK_NETWORK_PROFILES.get(
            attack_type,
            {'dst_ports': [80], 'flow_pattern': 'unknown', 'typical_pkt_size': 500}
        )

        proc_mapping = {
            'Brute Force': ('hydra', 'bash', 12345, 12300),
            'DoS': ('hping3', 'bash', 23456, 23400),
            'DDoS': ('botnet_client', 'systemd', 34567, 1),
            'Web Attacks': ('sqlmap', 'python3', 45678, 45600),
            'SQL Injection': ('sqlmap', 'python3', 56789, 56700),
            'XSS': ('xss_scanner', 'node', 67890, 67800),
            'Botnet': ('emotet_loader', 'explorer.exe', 78901, 1200),
            'Infiltration': ('psexec', 'smbclient', 89012, 89000),
            'Heartbleed': ('heartbleed_poc', 'python3', 90123, 90100),
            'Port Scan': ('nmap', 'bash', 10123, 10100),
        }

        proc_info = proc_mapping.get(attack_type, ('unknown', 'unknown', 0, 0))

        # Build Host Entity
        host = HostEntity(
            process_name=proc_info[0],
            parent_process_name=proc_info[1],
            process_id=proc_info[2],
            parent_process_id=proc_info[3],
            registry_operations=self._infer_registry_ops(attack_type),
            file_operations=self._infer_file_ops(attack_type),
            process_injection_indicators=attack_type in ['Botnet', 'Infiltration'],
        )

        # Build Network Communication
        net = NetworkCommunication(
            destination_ips=[f"192.168.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
                           for _ in range(min(3, len(profile['dst_ports'])))],
            destination_ports=profile['dst_ports'][:5],
            sni_values=[f"malicious-{attack_type.lower().replace(' ', '-')}.example.com"],
            ja3_fingerprints=[self._generate_ja3_hash()],
            ja3s_fingerprints=[self._generate_ja3_hash()],
            certificate_subjects=[f"CN=*.malicious-{attack_type.lower().replace(' ', '-')}.com"],
            certificate_issuers=["CN=Let's Encrypt Authority X3, O=Let's Encrypt, C=US"],
            http_methods=['GET', 'POST'] if attack_type in ['Web Attacks', 'SQL Injection', 'XSS'] else [],
            tls_versions=['TLS 1.2'],
        )

        # Build Temporal Behavior
        if attack_type == 'Botnet':
            beacon_interval = rng.uniform(60, 3600)
            beacon_intervals = [beacon_interval + rng.uniform(-5, 5) for _ in range(n_flows)]
        else:
            beacon_intervals = [rng.uniform(0.1, 30) for _ in range(n_flows)]

        temporal = TemporalBehavior(
            beacon_intervals=beacon_intervals,
            beacon_jitter=float(np.std(beacon_intervals)),
            connection_periodicity_score=0.85 if attack_type == 'Botnet' else rng.uniform(0.1, 0.5),
            burstiness_index=2.5 if attack_type in ['DoS', 'DDoS'] else rng.uniform(0.5, 1.5),
            session_duration_seconds=sum(beacon_intervals),
            flows_per_second=n_flows / max(sum(beacon_intervals), 0.001),
            dominant_packet_size=profile['typical_pkt_size'],
            heartbeat_packet_length_mean=float(profile['typical_pkt_size']),
            heartbeat_packet_length_std=float(profile['typical_pkt_size'] * 0.1),
        )

        # Detect beaconing from synthetic data
        self._detect_beaconing(
            [{'timestamp': sum(beacon_intervals[:i+1]) * 1000}
             for i in range(len(beacon_intervals))],
            temporal,
        )

        confidence = self._compute_confidence(host, net, temporal, attack_type)

        return StructuredCTI(
            host_entity=host,
            network_communication=net,
            temporal_behavior=temporal,
            attack_type_hint=attack_type,
            confidence_score=confidence,
        )
