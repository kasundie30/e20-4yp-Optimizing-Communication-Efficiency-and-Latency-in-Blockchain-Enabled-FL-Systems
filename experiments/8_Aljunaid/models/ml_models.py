"""
ml_models.py — Sklearn model factories for Aljunaid et al. (2025).

Paper Table 6 reports GBM, SVM, and LR results.
GBM achieved best performance (99.95% accuracy) and is used as the primary
model selected for global aggregation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def make_gbm():
    """Gradient Boosting Machine — best model per paper Table 6."""
    return GradientBoostingClassifier(
        n_estimators=config.GBM_N_ESTIMATORS,
        max_depth=config.GBM_MAX_DEPTH,
        learning_rate=config.GBM_LEARNING_RATE,
        random_state=config.RANDOM_SEED,
        subsample=0.8,
    )


def make_svm():
    """SVM with probability calibration (needed for predict_proba)."""
    base = SVC(
        C=config.SVM_C,
        kernel="rbf",
        probability=True,
        max_iter=config.SVM_MAX_ITER,
        random_state=config.RANDOM_SEED,
    )
    return base


def make_lr():
    """Logistic Regression."""
    return LogisticRegression(
        C=config.LR_C,
        max_iter=config.LR_MAX_ITER,
        solver="lbfgs",
        random_state=config.RANDOM_SEED,
        class_weight="balanced",
    )


MODEL_FACTORIES = {
    "GBM": make_gbm,
    "SVM": make_svm,
    "LR":  make_lr,
}


def build_model(name: str):
    """Return a fresh unfitted model by name."""
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name]()
