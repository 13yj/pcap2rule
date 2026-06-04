"""XGBoost attack type classifier training (Section III-D).

Trains an XGBoost classifier on 38-dim flow features to predict
the attack type distribution p ∈ R^K, which is then used as the
conditioning vector for the RAG retriever's type embedding.

Hyperparameters (paper defaults):
  - max_depth=6, n_estimators=100, learning_rate=0.1
  - Multi-class softprob objective (probability outputs)
  - 5-fold cross-validation for evaluation
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from ..utils.logging import get_logger
from ..utils.seed import set_seed


@dataclass
class XGBoostTrainingArgs:
    """Paper-specified XGBoost hyperparameters."""
    output_dir: str = './checkpoints/xgboost'
    max_depth: int = 6
    n_estimators: int = 100
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    objective: str = 'multi:softprob'
    eval_metric: str = 'mlogloss'
    n_folds: int = 5
    early_stopping_rounds: int = 10
    seed: int = 42


def _prepare_features(
    samples: List[Any],
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extract features and labels from dataset samples.

    Expects each sample to have flow_features (4, 38) and attack_type.

    Args:
        samples: List of TrafficSample or similar objects.

    Returns:
        (X, y, label_encoder) where X shape is (N, 4*38) = (N, 152).
    """
    X_list, y_list = [], []
    for sample in samples:
        if hasattr(sample, 'flow_features') and sample.flow_features is not None:
            ff = np.asarray(sample.flow_features, dtype=np.float32)
            X_list.append(ff.reshape(-1))  # (4, 38) -> (152,)
            y_list.append(getattr(sample, 'attack_type', 'Unknown'))

    X = np.stack(X_list)
    le = LabelEncoder()
    y = le.fit_transform(y_list)
    return X, y, le


def train_xgboost(
    train_samples: List[Any],
    val_samples: Optional[List[Any]] = None,
    args: Optional[XGBoostTrainingArgs] = None,
) -> Tuple[XGBClassifier, LabelEncoder, Dict[str, float]]:
    """Train XGBoost attack type classifier with cross-validation.

    Args:
        train_samples: Training samples with flow_features and attack_type.
        val_samples: Optional validation samples for hold-out evaluation.
        args: Training hyperparameters.

    Returns:
        (model, label_encoder, eval_metrics) tuple.
    """
    args = args or XGBoostTrainingArgs()
    logger = get_logger("pcap2rule.train_xgboost")
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, label_encoder = _prepare_features(train_samples)
    logger.info(f"Training data: X shape={X.shape}, classes={len(label_encoder.classes_)}")

    # Cross-validation
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )
    cv_scores = []

    best_model = None
    best_score = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            min_child_weight=args.min_child_weight,
            objective=args.objective,
            eval_metric=args.eval_metric,
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.seed,
            use_label_encoder=False,
            verbosity=0,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)
        logger.info(f"Fold {fold + 1}: accuracy={acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model

    # Hold-out evaluation
    eval_metrics = {'cv_mean_accuracy': np.mean(cv_scores), 'cv_std_accuracy': np.std(cv_scores)}

    if val_samples:
        X_test, y_test, _ = _prepare_features(val_samples)
        y_test_enc = label_encoder.transform([s.attack_type for s in val_samples])
        y_pred = best_model.predict(X_test)
        eval_metrics['holdout_accuracy'] = accuracy_score(y_test_enc, y_pred)

    logger.info(
        f"XGBoost training complete: CV accuracy = "
        f"{eval_metrics['cv_mean_accuracy']:.4f} +/- {eval_metrics['cv_std_accuracy']:.4f}"
    )

    # Save model
    model_path = os.path.join(args.output_dir, 'xgboost_model.pkl')
    le_path = os.path.join(args.output_dir, 'label_encoder.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    logger.info(f"Model saved to {model_path}")
    return best_model, label_encoder, eval_metrics


def load_xgboost(checkpoint_dir: str) -> Tuple[XGBClassifier, LabelEncoder]:
    """Load a trained XGBoost classifier.

    Args:
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        (model, label_encoder) tuple.
    """
    with open(os.path.join(checkpoint_dir, 'xgboost_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(checkpoint_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder
