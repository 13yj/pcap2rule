"""Dataset download script for CSE-CIC-IDS2018 and CIC-IDS2017.

The datasets are hosted by the Canadian Institute for Cybersecurity (CIC)
at the University of New Brunswick. Due to their size (multi-GB), this script
provides instructions and helper functions for downloading.

Usage:
    python scripts/download_datasets.py --dataset CSE-CIC-IDS2018 --output ./data
    python scripts/download_datasets.py --dataset CIC-IDS2017 --output ./data
    python scripts/download_datasets.py --dataset all --output ./data
"""

import os
import sys
import argparse
from pathlib import Path

# Known dataset URLs (may change; update as needed)
DATASET_URLS = {
    'CSE-CIC-IDS2018': {
        'url': 'https://www.unb.ca/cic/datasets/ids-2018.html',
        'description': 'CSE-CIC-IDS2018 — 10 days of network traffic with 7 attack families',
        'size': '~50 GB (compressed), ~120 GB (extracted)',
        'note': 'Requires manual download from the CIC website due to access policies.',
    },
    'CIC-IDS2017': {
        'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
        'description': 'CIC-IDS2017 — 5 days of network traffic with 6 attack families',
        'size': '~30 GB (compressed), ~50 GB (extracted)',
        'note': 'Requires manual download from the CIC website due to access policies.',
    },
}


def print_instructions(dataset: str, output_dir: str):
    """Print download instructions for a dataset."""
    info = DATASET_URLS.get(dataset)
    if not info:
        print(f"Unknown dataset: {dataset}")
        return

    target = Path(output_dir) / dataset
    print(f"""
{'='*70}
  {dataset} Download Instructions
{'='*70}
Description: {info['description']}
Size:        {info['size']}
Note:        {info['note']}

Download URL: {info['url']}

Steps:
  1. Visit the URL above in a web browser
  2. Complete any required registration (some datasets require it)
  3. Download the PCAP files for all days
  4. Extract to: {target / 'PCAPs'}
     Expected structure:
       {target}/
       └── PCAPs/
           ├── 2018-02-14/  (or Monday/ for IDS2017)
           ├── 2018-02-15/  (or Tuesday/)
           └── ...

  5. (Optional) Place labels.csv in {target}/labels.csv
     with columns: filename,label

  6. Verify: python -c "from pcap2rule.data_pipeline import {dataset.replace('-', '_')}; \\
        ds = {dataset.replace('-', '_')}('{target}'); samples = ds.load()"
{'='*70}
""")


def main():
    parser = argparse.ArgumentParser(
        description='Download instructions for Pcap2Rule datasets'
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['CSE-CIC-IDS2018', 'CIC-IDS2017', 'all'],
        help='Which dataset to download (default: all)',
    )
    parser.add_argument(
        '--output', type=str, default='./data',
        help='Output directory for datasets',
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in DATASET_URLS:
            print_instructions(ds, args.output)
    else:
        print_instructions(args.dataset, args.output)

    print("\nNote: Due to distribution policies, these datasets require manual download.")
    print("Once downloaded, update configs/data.yaml with the correct data_dir path.")


if __name__ == '__main__':
    main()
