import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_synthetic_data(num_samples=1000):
    """Generates a synthetic matrix representing proposal evaluation indicators."""
    np.random.seed(42)
    
    novelty_scores = np.random.uniform(20, 100, num_samples)
    
    financial_scores = np.random.uniform(30, 100, num_samples)
    
    budgets = np.random.uniform(50000, 5000000, num_samples)
    
    tech_alignment = np.random.uniform(0.1, 1.0, num_samples)
    

    scoring_matrix = (
        (novelty_scores * 0.4) + 
        (financial_scores * 0.3) + 
        (tech_alignment * 20) - 
        (budgets / 250000)
    )
    
    scoring_matrix += np.random.normal(0, 10, num_samples)
    
    approved = (scoring_matrix > 35).astype(int)
    
    df = pd.DataFrame({
        "novelty_score": novelty_scores,
        "financial_score": financial_scores,
        "budget": budgets,
        "tech_alignment": tech_alignment,
        "approved": approved
    })
    return df

def train_ensemble_pipeline():
    print(" Synthesizing analytical R&D training data...")
    data = generate_synthetic_data()
    
    X = data[["novelty_score", "financial_score", "budget", "tech_alignment"]]
    y = data["approved"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Ensemble Model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\n Model Evaluation Metrics Summary:")
    print(classification_report(y_test, predictions))
    
    os.makedirs("ml/artifacts", exist_ok=True)
    model_path = "ml/artifacts/ensemble_classifier.pkl"
    
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
        
    print(f"Success! Machine learning model serialized to: {model_path}\n")

if __name__ == "__main__":
    train_ensemble_pipeline()