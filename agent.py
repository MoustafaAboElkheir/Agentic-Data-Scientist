"""
Agentic Data Scientist
======================
An autonomous AI agent with planner, tools, memory, and reflection components.
It profiles a CSV dataset, selects preprocessing and modelling strategies,
evaluates results, re-plans when performance is below threshold, and persists
reproducible artefacts (cleaned data, trained model, JSONL experiment log).

Usage
-----
    python agent.py --dataset data/breast_cancer.csv --target target

Dependencies
------------
    pip install -r requirements.txt
"""
import os
import json
import logging
import argparse
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

from tools.eda_tools import profile_dataset, detect_target_column, summarize_class_balance
from tools.model_tools import preprocess, select_and_train, evaluate, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PERFORMANCE_THRESHOLD = 0.90   # AUC-ROC below this triggers re-planning
STRATEGY_FALLBACK_ORDER = ["xgboost", "random_forest", "svm", "logistic_regression"]


class AgenticDataScientist:
    """
    Closed-loop autonomous data science agent.

    Components
    ----------
    Planner   — decides which model strategy to try next
    Tools     — EDA, preprocessing, training, evaluation (see tools/)
    Memory    — list of past experiment results
    Reflector — inspects results and decides whether to re-plan
    """

    def __init__(self, performance_threshold: float = PERFORMANCE_THRESHOLD):
        self.threshold = performance_threshold
        self.memory: list[dict] = []
        self._strategy_queue = list(STRATEGY_FALLBACK_ORDER)
        logger.info("AgenticDataScientist initialised (threshold=%.2f)", self.threshold)

    # ── Planner ───────────────────────────────────────────────────────────────
    def _plan(self) -> str:
        if not self._strategy_queue:
            logger.warning("All strategies exhausted; falling back to xgboost")
            return "xgboost"
        strategy = self._strategy_queue.pop(0)
        logger.info("Planner selected strategy: %s", strategy)
        return strategy

    # ── Reflector ─────────────────────────────────────────────────────────────
    def _reflect(self, metrics: dict) -> bool:
        """Return True if re-planning is needed."""
        auc = metrics.get("auc_roc", 0)
        if auc < self.threshold:
            logger.warning("AUC-ROC %.4f < threshold %.2f → re-planning", auc, self.threshold)
            return True
        logger.info("AUC-ROC %.4f ≥ threshold %.2f → satisfied", auc, self.threshold)
        return False

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self, dataset_path: str, target_col: str = None) -> dict:
        logger.info("═" * 60)
        logger.info("Starting autonomous pipeline for: %s", dataset_path)

        # Step 1 — Profile
        df = pd.read_csv(dataset_path)
        profile = profile_dataset(df)
        logger.info("Profile: %d rows × %d cols | missing: %s",
                    profile["n_rows"], profile["n_cols"],
                    {k: v for k, v in profile["missing_values"].items() if v > 0} or "none")

        # Step 2 — Detect target
        if target_col is None:
            target_col = detect_target_column(df)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        balance_info = summarize_class_balance(y)
        logger.info("Class balance: %s", balance_info["percentages"])

        # Step 3 — Split (70 / 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # Step 4 — Agentic loop
        best_result = None
        while True:
            strategy = self._plan()

            # Preprocess (balance training set, scale features)
            X_tr_s, X_te_s, y_tr_bal, scaler = preprocess(X_train, X_test, y_train, balance=True)

            # Train
            model, used_strategy = select_and_train(X_tr_s, y_tr_bal, strategy=strategy)

            # Evaluate
            metrics = evaluate(model, X_te_s, y_test)
            logger.info("Results [%s] → Acc=%.4f | F1=%.4f | AUC=%.4f",
                        used_strategy, metrics["accuracy"], metrics["f1"], metrics["auc_roc"])

            # Record in memory
            record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "strategy": used_strategy,
                "metrics": {k: v for k, v in metrics.items() if k != "report"},
            }
            self.memory.append(record)

            # Reflect
            needs_replan = self._reflect(metrics)
            if not needs_replan or not self._strategy_queue:
                best_result = {"strategy": used_strategy, "metrics": metrics, "model": model}
                break

        # Step 5 — Persist artefacts
        os.makedirs("results", exist_ok=True)
        log_path = "results/experiment_log.jsonl"
        with open(log_path, "w") as f:
            for entry in self.memory:
                f.write(json.dumps(entry) + "\n")
        logger.info("Experiment log saved → %s", log_path)

        logger.info("═" * 60)
        logger.info("FINAL: strategy=%s | AUC=%.4f | Acc=%.4f | F1=%.4f",
                    best_result["strategy"],
                    best_result["metrics"]["auc_roc"],
                    best_result["metrics"]["accuracy"],
                    best_result["metrics"]["f1"])
        return best_result


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Data Scientist")
    parser.add_argument("--dataset", default="data/breast_cancer.csv")
    parser.add_argument("--target",  default="target")
    args = parser.parse_args()

    agent = AgenticDataScientist()
    result = agent.run(args.dataset, args.target)
    print("\n" + "=" * 50)
    print(json.dumps({k: v for k, v in result.items() if k != "model"}, indent=2))
