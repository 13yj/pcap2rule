"""CIC-IDS2017 dataset loader.

Dataset characteristics:
- 5 days of capture (July 3–7, 2017)
- 6 attack families: Brute Force, DoS, Web Attacks, Infiltration, Botnet, Port Scan
- Train/test split: Days 1–4 for training, Day 5 for testing (per paper Section IV-A)
"""

import os
import csv
import glob
from pathlib import Path
from typing import List

from .dataset_base import BaseDataset, TrafficSample


class CIC_IDS2017(BaseDataset):
    """Loader for the CIC-IDS2017 dataset.

    Expected directory structure:
        data_dir/
        ├── PCAPs/
        │   ├── Monday/     (Day 1 — training)
        │   ├── Tuesday/    (Day 2 — training)
        │   ├── Wednesday/  (Day 3 — training)
        │   ├── Thursday/   (Day 4 — training)
        │   └── Friday/     (Day 5 — testing)
        └── labels.csv
    """

    TRAIN_DATES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    TEST_DATES = ['Friday']

    ATTACK_TYPES = [
        'Brute Force', 'DoS', 'Web Attack', 'Infiltration',
        'Botnet', 'Port Scan',
    ]

    LABEL_MAP = {
        'BENIGN': 'benign',
        'Benign': 'benign',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Brute Force': 'Brute Force',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS': 'DoS',
        'Web Attack Brute Force': 'Web Attack',
        'Web Attack SQL Injection': 'Web Attack',
        'Web Attack XSS': 'Web Attack',
        'Web Attack': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Bot': 'Botnet',
        'Botnet': 'Botnet',
        'PortScan': 'Port Scan',
        'Port Scan': 'Port Scan',
    }

    def __init__(self, data_dir: str, seed: int = 42):
        super().__init__(data_dir, seed, self.ATTACK_TYPES)
        self.pcap_dir = self.data_dir / 'PCAPs'
        self.label_file = self.data_dir / 'labels.csv'

    def _parse_label(self, raw_label: str) -> str:
        return self.LABEL_MAP.get(raw_label.strip(), 'benign')

    def _load_sample_list(self) -> List[TrafficSample]:
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

            rel_path = os.path.relpath(pcap_path, self.pcap_dir)
            parts = rel_path.replace('\\', '/').split('/')
            day = parts[0] if len(parts) > 1 else 'unknown'

            attack_type = label_dict.get(fname, 'unknown')
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
                metadata={'day': day, 'date': day, 'filename': fname},
            ))

        print(f"CIC-IDS2017: loaded {len(samples)} PCAP samples "
              f"from {self.pcap_dir}")
        return samples

    def _load_labels(self) -> dict:
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
        """Return (train_samples, test_samples) using day-based split."""
        all_samples = self.load()
        train, test = [], []
        for s in all_samples:
            day = s.metadata.get('day', '')
            if any(d in day for d in self.TRAIN_DATES):
                train.append(s)
            elif any(d in day for d in self.TEST_DATES):
                test.append(s)
            else:
                train.append(s)
        print(f"IDS2017 split: train={len(train)}, test={len(test)}")
        return train, test
