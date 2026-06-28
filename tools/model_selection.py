"""
Model Selection and Hyperparameter Strategy
============================================
Defines the full model registry, hyperparameter grids, and selection logic
used by the Agentic Data Scientist pipeline.
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Full model registry with default hyperparameters
MODEL_REGISTRY = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "description": "Fast, interpretable baseline for linearly separable problems.",
        "complexity": "low",
    },
    "random_forest": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
        "description": "Ensemble of decision trees; robust to outliers and high-dimensional data.",
        "complexity": "medium",
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        "description": "Sequential boosting; high accuracy on tabular data.",
        "complexity": "medium",
    },
    "svm": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "description": "Effective in high-dimensional spaces; uses RBF kernel by default.",
        "complexity": "medium",
    },
    "xgboost": {
        "model": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "description": "Gradient-boosted trees; state-of-the-art on structured/tabular data.",
        "complexity": "high",
    },
}

# Strategy priority queue (agent tries these in order)
STRATEGY_PRIORITY = ["xgboost", "random_forest", "gradient_boosting", "svm", "logistic_regression"]


def select_strategy_by_profile(profile: dict) -> str:
    """
    Select an initial model strategy based on dataset profile.

    Rules
    -----
    - Small dataset (< 500 rows)  → logistic_regression (avoids overfitting)
    - High dimensionality (> 50 cols) → svm (handles high-dim well)
    - Default                     → xgboost (best general-purpose)
    """
    n_rows = profile.get("n_rows", 1000)
    n_cols = profile.get("n_cols", 10)

    if n_rows < 500:
        logger.info("Small dataset detected → selecting logistic_regression")
        return "logistic_regression"
    if n_cols > 50:
        logger.info("High-dimensional dataset detected → selecting svm")
        return "svm"
    logger.info("Default strategy → selecting xgboost")
    return "xgboost"


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """Run k-fold cross-validation and return mean/std of key metrics."""
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    f1_scores  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    return {
        "cv_auc_mean":  round(float(np.mean(auc_scores)), 4),
        "cv_auc_std":   round(float(np.std(auc_scores)), 4),
        "cv_f1_mean":   round(float(np.mean(f1_scores)), 4),
        "cv_f1_std":    round(float(np.std(f1_scores)), 4),
    }
