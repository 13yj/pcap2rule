"""Flow-level statistical feature extractor using nfstream.

Extracts 38-dimensional feature vectors for each bidirectional flow, organized
into 4 categories (Section III-C.1, Eq.1):
  - Temporal (10-dim): duration, IAT statistics (forward + backward)
  - Volumetric (8-dim): packet/byte counts and ratios
  - Flag-based (12-dim): TCP flag distributions
  - Protocol (8-dim): App protocol identification, transport protocol, service port

Output format per PCAP: (4, 38) tensor — 4 summary statistics (mean, std, max, min)
across all flows in the segment, per feature dimension.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from nfstream import NFStreamer
    NFSTREAM_AVAILABLE = True
except ImportError:
    NFSTREAM_AVAILABLE = False


class FlowFeatureExtractor:
    """Extract 38-dim flow-level statistical features using nfstream.

    Args:
        idle_timeout: Flow idle timeout in seconds.
        active_timeout: Flow active timeout in seconds.
        statistical_analysis: Whether to compute nfstream statistical features.
        n_dissections: Number of packets to dissect per flow (0=all, or limit).
    """

    # Feature dimension organization
    TEMPORAL_DIMS = [
        'bidirectional_duration_ms',
        'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
        'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
        'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
        'bidirectional_mean_iat', 'bidirectional_stddev_iat',
        'bidirectional_min_iat',
    ]

    VOLUMETRIC_DIMS = [
        'src2dst_packets', 'dst2src_packets',
        'src2dst_bytes', 'dst2src_bytes',
        'src2dst_psh_packets', 'dst2src_psh_packets',
        'bidirectional_packets', 'bidirectional_bytes',
    ]

    FLAG_DIMS = [
        'src2dst_syn_packets', 'dst2src_syn_packets',
        'src2dst_fin_packets', 'dst2src_fin_packets',
        'src2dst_rst_packets', 'dst2src_rst_packets',
        'src2dst_ack_packets', 'dst2src_ack_packets',
        'src2dst_urg_packets', 'dst2src_urg_packets',
        'src2dst_cwr_packets', 'dst2src_cwr_packets',
    ]

    # Protocol/app identification features are extracted separately
    # via nfstream's nDPI integration (nDPId)

    def __init__(
        self,
        idle_timeout: int = 120,
        active_timeout: int = 1800,
        statistical_analysis: bool = True,
        n_dissections: int = 20,
    ):
        if not NFSTREAM_AVAILABLE:
            raise ImportError(
                "nfstream is required for flow extraction. "
                "Install with: pip install nfstream"
            )
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.statistical_analysis = statistical_analysis
        self.n_dissections = n_dissections

    @property
    def feature_dim(self) -> int:
        return 38

    def extract(self, pcap_path: str) -> np.ndarray:
        """Extract per-flow features from a PCAP file and aggregate to segment level.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            np.ndarray of shape (4, 38): 4 summary statistics × 38 feature dimensions.
            Rows: [mean, std, max, min] across flows.
        """
        flows = self._extract_flows(pcap_path)
        if not flows:
            return np.zeros((4, self.feature_dim), dtype=np.float32)

        flow_matrix = self._flows_to_matrix(flows)
        return self._aggregate(flow_matrix)

    def _extract_flows(self, pcap_path: str) -> list:
        """Run nfstream on a PCAP and return flow objects."""
        streamer = NFStreamer(
            source=pcap_path,
            idle_timeout=self.idle_timeout,
            active_timeout=self.active_timeout,
            statistical_analysis=self.statistical_analysis,
            n_dissections=self.n_dissections,
        )
        return list(streamer)

    def _flows_to_matrix(self, flows: list) -> np.ndarray:
        """Convert nfstream flow objects to a (N_flows, 38) feature matrix.

        Uses standardized nfstream attribute names. Falls back to synthetic
        features when specific attributes are unavailable.
        """
        n_flows = len(flows)
        matrix = np.zeros((n_flows, self.feature_dim), dtype=np.float32)

        for i, flow in enumerate(flows):
            feats = []

            # --- Temporal (10-dim) ---
            # Flow duration and timing
            dur = getattr(flow, 'bidirectional_duration_ms', 0.0) or 0.0
            feats.append(dur)
            feats.append(getattr(flow, 'bidirectional_first_seen_ms', 0.0) or 0.0)
            feats.append(getattr(flow, 'bidirectional_last_seen_ms', 0.0) or 0.0)

            # Directional timing
            feats.append(getattr(flow, 'src2dst_first_seen_ms', 0.0) or 0.0)
            feats.append(getattr(flow, 'src2dst_last_seen_ms', 0.0) or 0.0)
            feats.append(getattr(flow, 'dst2src_first_seen_ms', 0.0) or 0.0)
            feats.append(getattr(flow, 'dst2src_last_seen_ms', 0.0) or 0.0)

            # IAT statistics
            iat_mean = getattr(flow, 'bidirectional_mean_iat', 0.0) or 0.0
            iat_std = getattr(flow, 'bidirectional_stddev_iat', 0.0) or 0.0
            iat_min = getattr(flow, 'bidirectional_min_iat', 0.0) or 0.0
            feats.extend([iat_mean, iat_std, iat_min])

            # --- Volumetric (8-dim) ---
            feats.append(getattr(flow, 'src2dst_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_bytes', 0) or 0)
            feats.append(getattr(flow, 'dst2src_bytes', 0) or 0)
            feats.append(getattr(flow, 'src2dst_psh_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_psh_packets', 0) or 0)
            feats.append(getattr(flow, 'bidirectional_packets', 0) or 0)
            feats.append(getattr(flow, 'bidirectional_bytes', 0) or 0)

            # --- Flag-based (12-dim) ---
            feats.append(getattr(flow, 'src2dst_syn_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_syn_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_fin_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_fin_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_rst_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_rst_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_ack_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_ack_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_urg_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_urg_packets', 0) or 0)
            feats.append(getattr(flow, 'src2dst_cwr_packets', 0) or 0)
            feats.append(getattr(flow, 'dst2src_cwr_packets', 0) or 0)

            # --- Protocol (8-dim) — one-hot encoded app protocol ---
            # nfstream provides application_protocol as an integer ID
            # and application_name as a string
            proto_id = getattr(flow, 'application_protocol', 0) or 0
            # Encode protocol ID and category into 8 features:
            # features: protocol_id, is_http, is_dns, is_tls, is_ssh, is_smtp, is_ftp, transport_proto
            app_name = str(getattr(flow, 'application_name', '')).lower()
            feats.append(float(proto_id))
            feats.append(1.0 if 'http' in app_name else 0.0)
            feats.append(1.0 if 'dns' in app_name else 0.0)
            feats.append(1.0 if 'tls' in app_name or 'ssl' in app_name else 0.0)
            feats.append(1.0 if 'ssh' in app_name else 0.0)
            feats.append(1.0 if 'smtp' in app_name else 0.0)
            feats.append(1.0 if 'ftp' in app_name else 0.0)
            # Transport protocol (TCP=6, UDP=17)
            proto = getattr(flow, 'protocol', 6) or 6
            feats.append(float(proto))

            matrix[i, :] = np.array(feats, dtype=np.float32)

        return matrix

    def _aggregate(self, flow_matrix: np.ndarray) -> np.ndarray:
        """Aggregate per-flow features to segment-level via 4 summary statistics.

        Args:
            flow_matrix: (N_flows, 38) feature matrix.

        Returns:
            (4, 38) tensor: [mean, std, max, min] per feature dimension.
        """
        if flow_matrix.shape[0] == 1:
            # Single flow — repeat
            return np.tile(flow_matrix, (4, 1))

        return np.stack([
            flow_matrix.mean(axis=0),   # μ_i
            flow_matrix.std(axis=0),    # σ_i
            flow_matrix.max(axis=0),    # max_i
            flow_matrix.min(axis=0),    # min_i
        ], axis=0).astype(np.float32)

    def extract_raw(self, pcap_path: str) -> List[dict]:
        """Extract per-flow features and return as list of dicts for inspection.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            List of flow feature dicts.
        """
        flows = self._extract_flows(pcap_path)
        result = []
        for flow in flows:
            result.append({
                'duration_ms': getattr(flow, 'bidirectional_duration_ms', 0),
                'src_packets': getattr(flow, 'src2dst_packets', 0),
                'dst_packets': getattr(flow, 'dst2src_packets', 0),
                'src_bytes': getattr(flow, 'src2dst_bytes', 0),
                'dst_bytes': getattr(flow, 'dst2src_bytes', 0),
                'total_packets': getattr(flow, 'bidirectional_packets', 0),
                'total_bytes': getattr(flow, 'bidirectional_bytes', 0),
                'app_protocol': getattr(flow, 'application_name', 'Unknown'),
                'protocol': getattr(flow, 'protocol', 0),
                'src_ip': getattr(flow, 'src_ip', ''),
                'dst_ip': getattr(flow, 'dst_ip', ''),
                'src_port': getattr(flow, 'src_port', 0),
                'dst_port': getattr(flow, 'dst_port', 0),
            })
        return result
