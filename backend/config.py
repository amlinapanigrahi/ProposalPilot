NOVELTY_WEIGHT = 0.4
FINANCIAL_WEIGHT = 0.35
TECHNICAL_WEIGHT = 0.25

# Single source of truth for feature column order. 

FEATURE_ORDER = ["novelty_score", "financial_score", "budget", "tech_alignment"]

# Threshold used only for synthetic label generation in train_model.py.
SYNTHETIC_APPROVAL_THRESHOLD = 35

# Below this predicted confidence, force human review regardless of the
# model's predicted class — avoids trusting coin-flip-level predictions.
MIN_CONFIDENCE_FOR_AUTO_DECISION = 60.0

MODEL_PATH = "ml/artifacts/ensemble_classifier.pkl"
PAST_PROJECTS_CSV_PATH = "data/past_projects.csv"