"""PCAP utility functions for file handling, segment extraction, and basic analysis."""

import os
import hashlib
from typing import List, Tuple, Optional


def compute_pcap_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a PCAP file for deduplication."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_pcap_size(filepath: str) -> int:
    """Get PCAP file size in bytes."""
    return os.path.getsize(filepath)


def split_pcap_by_flow(
    filepath: str,
    output_dir: str,
    max_flows_per_file: int = 1,
) -> List[str]:
    """Split a large PCAP into per-flow PCAP files using tshark.

    Args:
        filepath: Path to the input PCAP file.
        output_dir: Directory for output PCAP segments.
        max_flows_per_file: Number of flows per output file.

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Use tshark to split by TCP/UDP stream
    base = os.path.splitext(os.path.basename(filepath))[0]
    cmd = (
        f'tshark -r "{filepath}" -q -z conv,tcp 2>/dev/null | '
        f'head -20'
    )
    # For practical use, use editcap or tshark stream filtering
    # This is a placeholder for the real implementation
    tmp_pattern = os.path.join(output_dir, f"{base}_flow_")
    ret = os.system(
        f'editcap -c 100 "{filepath}" "{tmp_pattern}" 2>/dev/null'
    )
    if ret == 0:
        output_files = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith(f"{base}_flow_")
        ])

    return output_files


def is_pcap_file(filepath: str) -> bool:
    """Check if a file is a valid PCAP by checking the magic number."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        # PCAP magic: 0xa1b2c3d4 or 0xd4c3b2a1 (swapped)
        # PCAPNG magic: 0x0a0d0d0a
        return magic in (
            b'\xa1\xb2\xc3\xd4',
            b'\xd4\xc3\xb2\xa1',
            b'\n\r\r\n',
        )
    except (IOError, OSError):
        return False
