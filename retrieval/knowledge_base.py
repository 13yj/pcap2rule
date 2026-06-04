"""Suricata rule knowledge base — storage, loading, and management.

Maintains the curated knowledge base KB = {(r_i, e_type_i, e_sig_i)} as
described in Section III-D.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..utils.suricata_utils import SuricataRule, parse_suricata_rule


class KnowledgeBase:
    """Curated knowledge base of Suricata rules with dual embeddings.

    Args:
        kb_dir: Directory for knowledge base storage.
    """

    def __init__(self, kb_dir: str = "./knowledge_base"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.rules: List[SuricataRule] = []
        self.type_embeddings: Optional[np.ndarray] = None   # (N, 768)
        self.sig_embeddings: Optional[np.ndarray] = None     # (N, 384)
        self._rule_index: Dict[int, int] = {}  # sid -> position in rules list

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def is_indexed(self) -> bool:
        return (
            self.type_embeddings is not None
            and self.sig_embeddings is not None
            and len(self.rules) > 0
        )

    def load_rules(self, rules_dir: str = None):
        """Load Suricata rules from a directory of .rules files.

        Args:
            rules_dir: Directory containing .rules files. If None, uses kb_dir.
        """
        source = Path(rules_dir) if rules_dir else self.kb_dir
        for rules_file in source.glob("*.rules"):
            self._load_rules_file(rules_file)

        # Also try loading individual .rule files
        for rule_file in source.glob("*.rule"):
            self._load_rules_file(rule_file)

        # Load JSON rule database if exists
        json_path = self.kb_dir / "rules_db.json"
        if json_path.exists():
            self._load_json_db(json_path)

    def _load_rules_file(self, filepath: Path):
        """Parse a Suricata .rules file and add to KB."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                rule = parse_suricata_rule(line)
                if rule:
                    self.add_rule(rule)

    def _load_json_db(self, json_path: Path):
        """Load rules from a JSON knowledge base file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            rule_str = entry.get('rule', entry.get('rule_string', ''))
            if not rule_str:
                continue
            rule = parse_suricata_rule(rule_str)
            if rule:
                # Restore metadata
                if 'attack_type' in entry:
                    rule.metadata.append(f"attack_type:{entry['attack_type']}")
                self.add_rule(rule)

    def add_rule(self, rule: SuricataRule):
        """Add a single rule to the knowledge base."""
        if rule.sid in self._rule_index:
            idx = self._rule_index[rule.sid]
            self.rules[idx] = rule  # replace
        else:
            self._rule_index[rule.sid] = len(self.rules)
            self.rules.append(rule)

    def add_rules(self, rules: List[SuricataRule]):
        """Add multiple rules at once."""
        for rule in rules:
            self.add_rule(rule)

    def get_rule(self, sid: int) -> Optional[SuricataRule]:
        """Get a rule by its SID."""
        idx = self._rule_index.get(sid)
        if idx is not None:
            return self.rules[idx]
        return None

    def get_by_attack_type(self, attack_type: str) -> List[SuricataRule]:
        """Get all rules associated with a specific attack type."""
        results = []
        for rule in self.rules:
            for meta in rule.metadata:
                if f"attack_type:{attack_type}" in meta:
                    results.append(rule)
                    break
            # Also check msg field for type hints
            if attack_type.lower() in rule.msg.lower():
                if rule not in results:
                    results.append(rule)
        return results

    def set_embeddings(
        self,
        type_emb: np.ndarray,
        sig_emb: np.ndarray,
    ):
        """Set pre-computed embedding matrices.

        Args:
            type_emb: (N, 768) attack-type semantic embeddings.
            sig_emb: (N, 384) signature structure embeddings.
        """
        n = len(self.rules)
        if type_emb.shape[0] != n or sig_emb.shape[0] != n:
            raise ValueError(
                f"Embedding count mismatch: rules={n}, "
                f"type_emb={type_emb.shape[0]}, sig_emb={sig_emb.shape[0]}"
            )
        self.type_embeddings = type_emb.astype(np.float32)
        self.sig_embeddings = sig_emb.astype(np.float32)

    def save(self):
        """Persist the knowledge base to disk."""
        # Save rules as JSON
        rules_data = []
        for rule in self.rules:
            rules_data.append({
                'rule_string': rule.to_string(),
                'sid': rule.sid,
                'msg': rule.msg,
                'metadata': rule.metadata,
            })
        with open(self.kb_dir / "rules_db.json", 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        # Save embeddings
        if self.type_embeddings is not None:
            np.save(self.kb_dir / "type_embeddings.npy", self.type_embeddings)
        if self.sig_embeddings is not None:
            np.save(self.kb_dir / "sig_embeddings.npy", self.sig_embeddings)

    def load(self):
        """Load the knowledge base from disk."""
        json_path = self.kb_dir / "rules_db.json"
        if json_path.exists():
            self._load_json_db(json_path)

        type_path = self.kb_dir / "type_embeddings.npy"
        if type_path.exists():
            self.type_embeddings = np.load(type_path)

        sig_path = self.kb_dir / "sig_embeddings.npy"
        if sig_path.exists():
            self.sig_embeddings = np.load(sig_path)

    def get_rule_strings(self) -> List[str]:
        """Get all rules as serialized strings (for embedding generation)."""
        return [rule.to_string() for rule in self.rules]
