"""XGBoost-based attack type classifier (Section III-D).

Classifies attack type from flow features only, producing a probability
distribution over K attack types. Used to construct the type-weighted
query embedding for RAG retrieval.
"""

import os
import pickle
import numpy as np
from typing import List, Optional
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class AttackTypeClassifier:
    """Lightweight XGBoost classifier for attack type prediction.

    Trained on flow-level statistical features to predict the attack
    category (one of K types). The output probability distribution
    weights the type embedding query vector.

    Args:
        attack_types: List of attack type names (ordered for class indices).
        model_dir: Directory for saving trained models.
    """

    def __init__(
        self,
        attack_types: Optional[List[str]] = None,
        model_dir: str = "./checkpoints/xgboost",
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )
        self.attack_types = attack_types or []
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[xgb.XGBClassifier] = None
        self._label_to_idx: dict = {}
        self._idx_to_label: dict = {}

    @property
    def n_classes(self) -> int:
        return len(self.attack_types)

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
        max_depth: int = 6,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        """Train the XGBoost classifier.

        Args:
            X: (N, 152) feature matrix (4×38 flow features flattened).
            y: List of attack type labels.
            max_depth: Max tree depth.
            n_estimators: Number of boosting rounds.
            learning_rate: Step size shrinkage.
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio per tree.
        """
        # Build label mappings
        unique_labels = sorted(set(y))
        if not self.attack_types:
            self.attack_types = unique_labels

        self._label_to_idx = {lbl: i for i, lbl in enumerate(self.attack_types)}
        self._idx_to_label = {i: lbl for i, lbl in enumerate(self.attack_types)}

        y_int = np.array([self._label_to_idx.get(lbl, 0) for lbl in y])

        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
        self.model.fit(X, y_int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: (N, D) feature matrix.

        Returns:
            (N,) array of predicted attack type strings.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        y_int = self.model.predict(X)
        return np.array([self._idx_to_label.get(i, 'unknown') for i in y_int])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution.

        Args:
            X: (N, D) feature matrix.

        Returns:
            (N, K) probability matrix, one row per sample.
            Each row sums to 1.0.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def predict_single_proba(self, flow_features: np.ndarray) -> np.ndarray:
        """Predict probability distribution for a single sample.

        Args:
            flow_features: (4, 38) aggregated flow features.

        Returns:
            (K,) probability vector.
        """
        # Flatten (4, 38) → (152,)
        X = flow_features.flatten().reshape(1, -1)
        return self.predict_proba(X)[0]

    def save(self):
        """Save the trained model."""
        path = self.model_dir / "xgboost_attack_classifier.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'attack_types': self.attack_types,
                'label_to_idx': self._label_to_idx,
                'idx_to_label': self._idx_to_label,
            }, f)

    def load(self):
        """Load a trained model."""
        path = self.model_dir / "xgboost_attack_classifier.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.attack_types = data['attack_types']
        self._label_to_idx = data['label_to_idx']
        self._idx_to_label = data['idx_to_label']
