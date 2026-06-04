"""CSE-CIC-IDS2018 dataset loader.

Dataset characteristics:
- 10 days of capture (Feb 14 – Mar 2, 2018)
- 7 attack families: Brute Force, DoS, Web Attacks, Infiltration, Botnet, DDoS, Heartbleed
- Train/test split: Days 1–7 for training, Days 8–10 for testing (per paper Section IV-A)
"""

import os
import csv
import glob
from pathlib import Path
from typing import List, Optional

from .dataset_base import BaseDataset, TrafficSample


class CSE_CIC_IDS2018(BaseDataset):
    """Loader for the CSE-CIC-IDS2018 dataset.

    Expected directory structure:
        data_dir/
        ├── PCAPs/
        │   ├── 2018-02-14/  (Day 1 — training)
        │   ├── 2018-02-15/  (Day 2 — training)
        │   ├── ...
        │   ├── 2018-03-01/  (Day 8 — testing)
        │   └── 2018-03-02/  (Day 9 — testing)
        └── labels.csv       (optional: pre-computed labels)
    """

    # Paper-specified train/test date split
    TRAIN_DATES = [
        '2018-02-14', '2018-02-15', '2018-02-16',
        '2018-02-19', '2018-02-20', '2018-02-21',
        '2018-02-22',  # Days 1–7 (Feb 14–22, excluding weekend 17-18)
    ]
    TEST_DATES = [
        '2018-02-23', '2018-02-28', '2018-03-01', '2018-03-02',
    ]

    ATTACK_TYPES = [
        'Brute Force', 'DoS', 'Web Attack', 'Infiltration',
        'Botnet', 'DDoS', 'Heartbleed',
    ]

    # Label normalization map
    LABEL_MAP = {
        'Benign': 'benign',
        'FTP-BruteForce': 'Brute Force',
        'SSH-BruteForce': 'Brute Force',
        'Brute Force': 'Brute Force',
        'BruteForce': 'Brute Force',
        'DoS attacks-GoldenEye': 'DoS',
        'DoS attacks-Slowloris': 'DoS',
        'DoS attacks-SlowHTTPTest': 'DoS',
        'DoS attacks-Hulk': 'DoS',
        'DoS': 'DoS',
        'DDOS attack-HOIC': 'DDoS',
        'DDOS attack-LOIC-UDP': 'DDoS',
        'DDOS attack-LOIC-HTTP': 'DDoS',
        'DDoS': 'DDoS',
        'Web Attacks-Brute Force': 'Web Attack',
        'Web Attacks-Sql Injection': 'Web Attack',
        'Web Attacks-XSS': 'Web Attack',
        'Web Attack': 'Web Attack',
        'Infilteration': 'Infiltration',
        'Infiltration': 'Infiltration',
        'Bot': 'Botnet',
        'Botnet': 'Botnet',
        'Heartbleed': 'Heartbleed',
    }

    def __init__(self, data_dir: str, seed: int = 42):
        super().__init__(data_dir, seed, self.ATTACK_TYPES)
        self.pcap_dir = self.data_dir / 'PCAPs'
        self.label_file = self.data_dir / 'labels.csv'

    def _parse_label(self, raw_label: str) -> str:
        return self.LABEL_MAP.get(raw_label.strip(), 'benign')

    def _load_sample_list(self) -> List[TrafficSample]:
        """Scan the PCAPs directory and build the sample list.

        Each PCAP file corresponds to one attack or benign traffic segment.
        """
        samples = []
        pcap_files = glob.glob(
            str(self.pcap_dir / '**' / '*.pcap'), recursive=True
        ) + glob.glob(
            str(self.pcap_dir / '**' / '*.pcapng'), recursive=True
        )

        label_dict = self._load_labels() if self.label_file.exists() else {}

        for pcap_path in sorted(pcap_files):
            fname = os.path.basename(pcap_path)
            sample_id = os.path.splitext(fname)[0].replace(' ', '_')

            # Infer date from parent directory name
            rel_path = os.path.relpath(pcap_path, self.pcap_dir)
            parts = rel_path.replace('\\', '/').split('/')
            date = parts[0] if len(parts) > 1 else 'unknown'

            # Determine attack type from label map or directory structure
            attack_type = label_dict.get(fname, 'unknown')
            # Try to infer from directory name
            if attack_type == 'unknown':
                for part in parts:
                    part_lower = part.lower()
                    if 'benign' in part_lower:
                        attack_type = 'benign'
                        break
                    for atype in self.ATTACK_TYPES:
                        if atype.lower().replace(' ', '') in part_lower.replace(' ', ''):
                            attack_type = atype
                            break
                    if attack_type != 'unknown':
                        break

            samples.append(TrafficSample(
                sample_id=sample_id,
                pcap_path=pcap_path,
                attack_type=attack_type,
                metadata={'date': date, 'filename': fname},
            ))

        print(f"CSE-CIC-IDS2018: loaded {len(samples)} PCAP samples "
              f"from {self.pcap_dir}")
        return samples

    def _load_labels(self) -> dict:
        """Load label CSV if available."""
        labels = {}
        try:
            with open(self.label_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get('filename', row.get('File', row.get('file', '')))
                    label = row.get('label', row.get('Label', row.get('attack_type', '')))
                    if fname:
                        labels[fname] = label
        except Exception:
            pass
        return labels

    def get_train_test_split(self) -> tuple:
        """Return (train_samples, test_samples) using paper-specified date split.

        Returns:
            (train_samples, test_samples) tuple.
        """
        all_samples = self.load()
        train, test = [], []
        for s in all_samples:
            date = s.metadata.get('date', '')
            if any(d in date for d in self.TRAIN_DATES):
                train.append(s)
            elif any(d in date for d in self.TEST_DATES):
                test.append(s)
            else:
                # Assign to train by default
                train.append(s)
        print(f"IDS2018 split: train={len(train)}, test={len(test)}")
        return train, test
