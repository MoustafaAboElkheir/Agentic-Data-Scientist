# Agentic Data Scientist

An autonomous AI agent designed to profile datasets, select preprocessing strategies, train models, and evaluate results. Built with LangChain, local LLM inference (LLaVA/Ollama), and custom Python tools.

## Features
- **Autonomous Planning:** Analyzes raw CSV datasets and formulates a step-by-step data science plan.
- **Tool Orchestration:** Uses specialized tools for EDA, feature engineering, and model training (Random Forest, XGBoost, SVM).
- **Memory & Reflection:** Evaluates model performance (e.g., AUC-ROC) and autonomously re-plans if metrics fall below target thresholds.
- **Reproducibility:** Persists all artifacts, including cleaned datasets, trained models, and structured JSONL experiment logs.

## Architecture
The system implements a closed-loop control flow where the agent acts as the orchestrator:
1. **Profiler:** Extracts metadata (missing values, distributions).
2. **Planner:** Decides on imputation, scaling, and model selection.
3. **Executor:** Runs the PyTorch/scikit-learn pipelines.
4. **Reflector:** Analyzes outputs and adjusts hyperparameters.

## Setup
```bash
pip install -r requirements.txt
python agent.py --dataset data/sample.csv
```

## Technologies
- Python, LangChain, Ollama
- scikit-learn, Pandas, NumPy
- PyTorch

*Created by Moustafa AbouElkheir*
