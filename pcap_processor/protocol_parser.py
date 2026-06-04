"""Protocol parser for HTTP and DNS payload content.

Provides specialized parsing for extracting structured fields from
application-layer protocol data — HTTP methods, URIs, headers, and
DNS query names/types.
"""

import re
from typing import Dict, Optional, List, Tuple


def parse_http_request_line(payload) -> Tuple[str, str, str]:
    """Parse the HTTP request line.

    Args:
        payload: Raw HTTP request (bytes or str).

    Returns:
        (method, uri, version) tuple.
    """
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8', errors='replace')
    text = payload
    first_line = text.split('\r\n')[0]
    parts = first_line.split(' ')
    method = parts[0] if len(parts) > 0 else 'GET'
    uri = parts[1] if len(parts) > 1 else '/'
    version = parts[2] if len(parts) > 2 else 'HTTP/1.1'
    return method, uri, version


def parse_http_headers(payload) -> Dict[str, str]:
    """Parse HTTP headers from raw request/response bytes.

    Returns:
        Dict of header name -> header value.
    """
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8', errors='replace')
    text = payload
    headers = {}
    lines = text.split('\r\n')[1:]  # skip request/status line
    for line in lines:
        if line == '':
            break  # end of headers
        if ':' in line:
            k, v = line.split(':', 1)
            headers[k.strip()] = v.strip()
    return headers


def extract_http_body(payload: bytes) -> bytes:
    """Extract HTTP body (everything after double CRLF)."""
    idx = payload.find(b'\r\n\r\n')
    if idx == -1:
        return b''
    return payload[idx + 4:]


def normalize_payload(text: str) -> str:
    """Normalize payload text for LLM consumption.

    - Collapse multiple whitespace
    - Strip non-printable chars
    - Truncate very long tokens
    """
    # Remove non-printable except common whitespace
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Collapse repeated characters (base64-like patterns)
    text = re.sub(r'([A-Za-z0-9+/])\1{20,}', r'\1' * 10 + '...', text)
    return text.strip()


def parse_dns_query(payload: bytes) -> Optional[Dict[str, object]]:
    """Parse a DNS query from raw bytes (basic implementation).

    Note: For production use, prefer scapy's DNS layer parsing.
    This is a fallback for simple cases.
    """
    if len(payload) < 12:
        return None

    # DNS header: ID(2) + Flags(2) + QDCOUNT(2) + ANCOUNT(2) + NSCOUNT(2) + ARCOUNT(2)
    txid = int.from_bytes(payload[0:2], 'big')
    qdcount = int.from_bytes(payload[4:6], 'big')

    # Parse question section
    pos = 12
    qname_parts = []
    while pos < len(payload) and payload[pos] != 0:
        length = payload[pos]
        pos += 1
        if pos + length <= len(payload):
            qname_parts.append(payload[pos:pos + length].decode('ascii', errors='replace'))
            pos += length
    pos += 1  # skip null terminator

    if pos + 4 > len(payload):
        return None
    qtype = int.from_bytes(payload[pos:pos + 2], 'big')
    qclass = int.from_bytes(payload[pos + 2:pos + 4], 'big')

    return {
        'transaction_id': txid,
        'query_name': '.'.join(qname_parts),
        'query_type': qtype,
        'query_class': qclass,
    }


def hex_decode_payload(hex_str: str) -> bytes:
    """Decode a hex string to raw bytes."""
    hex_str = hex_str.replace(' ', '').replace('\n', '')
    try:
        return bytes.fromhex(hex_str)
    except ValueError:
        return b''


def extract_malicious_tokens(text: str) -> List[str]:
    """Extract potentially malicious tokens from payload text.

    Used for highlighting attack-relevant content in the prompt.
    """
    patterns = [
        r'(?:UNION\s+(?:ALL\s+)?SELECT)',
        r'(?:SELECT\s+.+\s+FROM\s+.+)',
        r'(?:\bOR\s+1\s*=\s*1\b)',
        r'(?:\bAND\s+1\s*=\s*1\b)',
        r"['\"]\s*(?:OR|AND)\s",
        r'<script[^>]*>',
        r'(?:onerror|onload|onclick)\s*=',
        r'(?:\.\.\/|\.\.\\){2,}',  # path traversal
        r'(?:/etc/passwd|/bin/sh|cmd\.exe)',
        r'(?:\bwget\s+|curl\s+)\S+',
        r'(?:eval\s*\(|exec\s*\()',
        r'%[0-9A-Fa-f]{2}',  # URL encoding
        r'(?:\\x[0-9A-Fa-f]{2})+',  # hex escape
    ]
    tokens = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            tokens.append(match.group())
    return list(dict.fromkeys(tokens))  # dedup, preserve order
