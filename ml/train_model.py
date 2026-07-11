import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from backend.config import (
    NOVELTY_WEIGHT,
    FINANCIAL_WEIGHT,
    TECHNICAL_WEIGHT,
    FEATURE_ORDER,
    SYNTHETIC_APPROVAL_THRESHOLD,
    MODEL_PATH,
)


def generate_synthetic_data(num_samples=1000):
    #Generates a synthetic matrix representing proposal evaluation indicators.
    
    np.random.seed(42)

    novelty_scores = np.random.uniform(20, 100, num_samples)
    financial_scores = np.random.uniform(30, 100, num_samples)
    budgets = np.random.uniform(50000, 5000000, num_samples)
    tech_alignment = np.random.uniform(10, 100, num_samples)

    quality_score = (
        (novelty_scores * NOVELTY_WEIGHT) +
        (financial_scores * FINANCIAL_WEIGHT) +
        (tech_alignment * TECHNICAL_WEIGHT)
    )

    budget_penalty = budgets / 250000

    scoring_matrix = quality_score - budget_penalty
    scoring_matrix += np.random.normal(0, 10, num_samples)

    approved = (scoring_matrix > SYNTHETIC_APPROVAL_THRESHOLD).astype(int)

    df = pd.DataFrame({
        "novelty_score": novelty_scores,
        "financial_score": financial_scores,
        "budget": budgets,
        "tech_alignment": tech_alignment,
        "approved": approved
    })
    return df


def train_ensemble_pipeline():
    print("Synthesizing analytical R&D training data...")
    data = generate_synthetic_data()

    X = data[FEATURE_ORDER]
    y = data["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Ensemble Model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nModel Evaluation Metrics Summary:")
    print(classification_report(y_test, predictions))
    print(
        "NOTE: High accuracy here reflects that labels are a deterministic "
        "function of the features plus noise (synthetic ground truth), not "
        "generalization to real-world approval patterns.\n"
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Success! Machine learning model serialized to: {MODEL_PATH}\n")


if __name__ == "__main__":
    train_ensemble_pipeline()