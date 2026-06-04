"""PCAP processor module — multi-modal feature extraction pipeline."""

from .flow_extractor import FlowFeatureExtractor
from .payload_extractor import PayloadExtractor
from .protocol_parser import (
    parse_http_request_line,
    parse_http_headers,
    extract_http_body,
    normalize_payload,
    parse_dns_query,
    hex_decode_payload,
    extract_malicious_tokens,
)
from .prompt_builder import PromptBuilder
from .cti_builder import CTIBuilder, StructuredCTI, HostEntity, NetworkCommunication, TemporalBehavior

__all__ = [
    'FlowFeatureExtractor',
    'PayloadExtractor',
    'parse_http_request_line',
    'parse_http_headers',
    'extract_http_body',
    'normalize_payload',
    'parse_dns_query',
    'hex_decode_payload',
    'extract_malicious_tokens',
    'PromptBuilder',
    'CTIBuilder',
    'StructuredCTI',
    'HostEntity',
    'NetworkCommunication',
    'TemporalBehavior',
]
