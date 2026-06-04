"""Payload extraction from PCAP using scapy.

Extracts application-layer payload text from HTTP, DNS, and other protocols.
For HTTP: parses method, URI, headers, and body.
For DNS: extracts query names and response data.
For other protocols: encodes first 512 bytes as hex string.
"""

from typing import Optional, List, Dict, Tuple
from collections import defaultdict

try:
    from scapy.all import rdpcap, Packet, IP, TCP, UDP, Raw
    from scapy.layers.http import HTTPRequest, HTTPResponse
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    Packet = object  # placeholder type for type annotations
    Raw = object
    IP = object
    TCP = object
    UDP = object


class PayloadExtractor:
    """Extract human-readable payload content from PCAP files.

    Args:
        max_payload_bytes: Maximum bytes to extract per protocol type.
        max_total_length: Maximum total output text length (characters).
    """

    HTTP_METHODS = {b'GET', b'POST', b'PUT', b'DELETE', b'HEAD',
                    b'OPTIONS', b'PATCH', b'CONNECT', b'TRACE'}

    def __init__(self, max_payload_bytes: int = 512, max_total_length: int = 2048):
        if not SCAPY_AVAILABLE:
            raise ImportError(
                "scapy is required for payload extraction. "
                "Install with: pip install scapy"
            )
        self.max_payload_bytes = max_payload_bytes
        self.max_total_length = max_total_length

    def extract(self, pcap_path: str) -> str:
        """Extract payload text from a PCAP file.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            String containing extracted payload text, truncated to max_total_length.
        """
        packets = self._read_packets(pcap_path)
        http_payloads = self._extract_http(packets)
        dns_payloads = self._extract_dns(packets)
        other_payloads = self._extract_other(packets)

        parts = []
        if http_payloads:
            parts.append("[HTTP PAYLOAD]")
            parts.extend(http_payloads[:5])  # max 5 HTTP exchanges
        if dns_payloads:
            parts.append("[DNS PAYLOAD]")
            parts.extend(dns_payloads[:5])
        if other_payloads:
            parts.append("[OTHER PAYLOAD (hex)]")
            parts.extend(other_payloads[:5])

        result = "\n".join(parts)
        if len(result) > self.max_total_length:
            result = result[:self.max_total_length - 3] + "..."
        return result

    def _read_packets(self, pcap_path: str) -> List[Packet]:
        """Read PCAP file and return list of packets."""
        try:
            packets = rdpcap(pcap_path)
        except Exception:
            return []
        return packets

    def _extract_http(self, packets: List[Packet]) -> List[str]:
        """Extract HTTP request/response payloads."""
        results = []
        http_pairs: Dict[int, dict] = defaultdict(dict)

        for pkt in packets:
            if not pkt.haslayer(TCP) or not pkt.haslayer(Raw):
                continue

            payload = bytes(pkt[Raw].load)
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport

            # Try HTTP Request
            try:
                if payload[:4] in [b'GET ', b'POST', b'PUT ', b'DELE',
                                   b'HEAD', b'OPTI', b'PATC', b'CONN']:
                    method, uri, headers, body = self._parse_http_request(payload)
                    results.append(
                        f"{method} {uri} HTTP/1.1\n"
                        + "\n".join(f"{k}: {v}" for k, v in headers.items())
                        + (f"\n\n[Body: {body[:200]}]" if body else "")
                    )
            except Exception:
                pass

            # Try HTTP Response
            try:
                if payload.startswith(b'HTTP/'):
                    status, headers, body = self._parse_http_response(payload)
                    results.append(
                        f"HTTP/1.1 {status}\n"
                        + "\n".join(f"{k}: {v}" for k, v in headers.items())
                        + (f"\n\n[Body: {body[:200]}]" if body else "")
                    )
            except Exception:
                pass

        return results

    def _parse_http_request(self, payload: bytes) -> Tuple[str, str, dict, bytes]:
        """Parse an HTTP request from raw bytes."""
        lines = payload.split(b'\r\n')
        first_line = lines[0].decode('utf-8', errors='replace')
        parts = first_line.split(' ')
        method = parts[0] if len(parts) > 0 else 'GET'
        uri = parts[1] if len(parts) > 1 else '/'

        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == b'':
                body_start = i + 1
                break
            if b':' in line:
                k, v = line.split(b':', 1)
                headers[k.decode('utf-8', errors='replace').strip()] = \
                    v.decode('utf-8', errors='replace').strip()

        body = b'\r\n'.join(lines[body_start:]) if body_start < len(lines) else b''
        return method, uri, headers, body[:self.max_payload_bytes]

    def _parse_http_response(self, payload: bytes) -> Tuple[str, dict, bytes]:
        """Parse an HTTP response from raw bytes."""
        lines = payload.split(b'\r\n')
        status = lines[0].decode('utf-8', errors='replace')
        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == b'':
                body_start = i + 1
                break
            if b':' in line:
                k, v = line.split(b':', 1)
                headers[k.decode('utf-8', errors='replace').strip()] = \
                    v.decode('utf-8', errors='replace').strip()

        body = b'\r\n'.join(lines[body_start:]) if body_start < len(lines) else b''
        return status, headers, body[:self.max_payload_bytes]

    def _extract_dns(self, packets: List[Packet]) -> List[str]:
        """Extract DNS query/response information."""
        results = []
        for pkt in packets:
            if not pkt.haslayer(UDP) or not pkt.haslayer(DNS):
                continue
            dns = pkt[DNS]
            qname = ''
            if dns.qd and hasattr(dns.qd, 'qname'):
                qname = dns.qd.qname.decode('utf-8', errors='replace') if isinstance(dns.qd.qname, bytes) else str(dns.qd.qname)
            qtype = dns.qd.qtype if dns.qd else 0
            results.append(f"DNS Query: {qname} (type={qtype})")

            # Extract response data if present
            if dns.an:
                for i in range(min(dns.ancount, 3)):
                    ans = dns.an[i] if hasattr(dns.an, '__getitem__') else dns.an
                    rdata = getattr(ans, 'rdata', '')
                    if isinstance(rdata, bytes):
                        rdata = rdata.hex()
                    results.append(f"  -> Answer: {rdata}")
        return results

    def _extract_other(self, packets: List[Packet]) -> List[str]:
        """Extract payload from non-HTTP, non-DNS packets as hex."""
        results = []
        extracted = 0
        target = min(self.max_payload_bytes, 200)

        for pkt in packets:
            if not pkt.haslayer(Raw):
                continue
            if pkt.haslayer(HTTPRequest) or pkt.haslayer(HTTPResponse) or pkt.haslayer(DNS):
                continue

            payload = bytes(pkt[Raw].load)
            if len(payload) == 0:
                continue

            hex_str = payload[:target].hex()
            proto = 'TCP' if pkt.haslayer(TCP) else 'UDP' if pkt.haslayer(UDP) else 'OTHER'
            sport = pkt[TCP].sport if pkt.haslayer(TCP) else pkt[UDP].sport if pkt.haslayer(UDP) else 0
            dport = pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport if pkt.haslayer(UDP) else 0

            results.append(f"[{proto}:{sport}>{dport}] {hex_str}")
            extracted += 1
            if extracted >= 3:
                break

        return results
