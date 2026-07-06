import os
import pickle
import numpy as np
from backend.services.financial_checker import FinancialChecker
from backend.services.document_parser import extract_text_from_pdf
from ml.vector_store import LocalVectorStore 
from backend.services.genai_narrative import GeminiExplainer

class EvaluationPipeline:
    def __init__(self, model_path: str = "ml/artifacts/ensemble_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        
        self.vector_store = LocalVectorStore()
        
        self._load_classifier()

    def _load_classifier(self):
        """Loads the serialized Random Forest ensemble model weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Machine Learning model artifact missing at {self.model_path}. "
                "Please run 'python -m ml.train_model' first to train the classifier."
            )
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def evaluate_incoming_proposal(self, text_content: str, budget: float) -> dict:
        """
        Runs the proposal through all analysis layers and outputs a final 
        predictive administrative recommendation.
        """
        novelty_score = self.vector_store.compute_novelty_score(text_content) 
        matched_projects = []

        if novelty_score >= 75:
            novelty_rating = "Highly Novel"
        elif novelty_score >= 45:
            novelty_rating = "Moderately Novel"
        else:
            novelty_rating = "Low Novelty / Potential Redundancy"

        financial_analysis = FinancialChecker.evaluate_budget_feasibility(budget, text_content)
        financial_score = financial_analysis["financial_score"]

        tech_keywords = ["algorithm", "optimization", "framework", "system", "neural", "database", "security", "pipeline"]
        lower_text = text_content.lower()
        matched_words = [word for word in tech_keywords if word in lower_text]
        tech_alignment = max(0.1, min(1.0, len(matched_words) / len(tech_keywords)))

        feature_vector = np.array([[
            novelty_score,
            financial_score,
            float(budget if budget else 0.0),
            tech_alignment
        ]])

        prediction_id = int(self.model.predict(feature_vector)[0])
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence_score = float(probabilities[prediction_id]) * 100

        final_decision = "RECOMMENDED FOR APPROVAL" if prediction_id == 1 else "FLAGGED FOR HUMAN REVIEW"

        extracted_themes = self.vector_store.extract_top_keywords(text_content, top_n=5)

        ai_narrative = GeminiExplainer.generate_narrative(
            proposal_title="Uploaded Document Asset",
            metrics={"novelty_score": novelty_score, "financial_score": financial_score},
            extracted_keywords=extracted_themes
        )

        importances = self.model.feature_importances_

        
        return {
        "status": "Success",
        "metrics": {
            "novelty_score": round(novelty_score, 2),
            "novelty_rating": novelty_rating,
            "financial_score": round(financial_score, 2),
            "financial_rating": financial_analysis["financial_rating"]
        },
        "explainable_ai_attribution": {
            "novelty_impact_weight": round(importances[0] * 100, 1),
            "financial_feasibility_weight": round(importances[1] * 100, 1),
            "technical_density_weight": round(importances[2] * 100, 1),
            "methodology_type": "Gini Importance / Mean Decrease in Impurity"
        },
        "ai_generated_narrative": ai_narrative,
        "similar_past_projects": matched_projects,
        "evaluation_verdict": {
            "decision": final_decision,
            "confidence_percentage": round(confidence_score, 2)
        }
}