"""
Model Tools for the Agentic Data Scientist pipeline.
Handles preprocessing, model selection, training, and evaluation.
"""
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "svm":                 SVC(probability=True, random_state=42),
    "xgboost":             XGBClassifier(n_estimators=200, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}


def preprocess(X_train, X_test, y_train, balance: bool = True):
    """Scale features and optionally balance the training set (70/30 split methodology)."""
    if balance:
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logger.info(f"After balancing — class distribution: {np.bincount(y_train)}")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, scaler


def select_and_train(X_train, y_train, strategy: str = "xgboost"):
    """Train a model from the registry given a strategy name."""
    if strategy not in MODEL_REGISTRY:
        logger.warning(f"Unknown strategy '{strategy}'; defaulting to 'xgboost'")
        strategy = "xgboost"
    model = MODEL_REGISTRY[strategy]
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {strategy}")
    return model, strategy


def evaluate(model, X_test, y_test) -> dict:
    """Return a comprehensive evaluation dictionary."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds, average='weighted'), 4),
        "auc_roc":  round(roc_auc_score(y_test, proba), 4),
        "report":   classification_report(y_test, preds, output_dict=True),
    }
