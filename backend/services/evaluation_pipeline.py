import os
import pickle
import numpy as np
from backend.services.financial_checker import FinancialChecker
from backend.services.document_parser import extract_text_from_pdf
from ml.vector_store import LocalVectorStore
from ml.shap_explainer import ShapExplainer
from backend.services.genai_narrative import GeminiExplainer
from backend.config import (
    NOVELTY_WEIGHT,
    FINANCIAL_WEIGHT,
    TECHNICAL_WEIGHT,
    FEATURE_ORDER,
    MIN_CONFIDENCE_FOR_AUTO_DECISION,
    MODEL_PATH,
)


class EvaluationPipeline:
    def __init__(self, model_path: str = MODEL_PATH):

        self.model_path = model_path
        self.model = None
        self.shap_explainer = None

        self.vector_store = LocalVectorStore()
        self._load_classifier()
        self.shap_explainer = ShapExplainer(self.model, FEATURE_ORDER)

    def _load_classifier(self):
        #Loads the serialized Random Forest ensemble model weights.
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Machine Learning model artifact missing."
            )
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def evaluate_incoming_proposal(
        self, text_content: str, budget: float, proposal_title: str = "Untitled Proposal") -> dict:
        """
        Runs the proposal through all analysis layers and outputs a final
        predictive administrative recommendation.
        """

        analysis = self.vector_store.analyze(text_content)

        novelty_score = analysis["novelty_score"]
        tech_alignment = analysis["technical_alignment"]
        matched_projects = analysis["similar_projects"]
        extracted_themes = analysis["top_keywords"]

        if novelty_score >= 75:
            novelty_rating = "Highly Novel"
        elif novelty_score >= 45:
            novelty_rating = "Moderately Novel"
        else:
            novelty_rating = "Low Novelty / Potential Redundancy"

        # Semantic financial analysis.
        financial_context_scores = self.vector_store.compute_financial_context_scores(text_content)
        financial_analysis = FinancialChecker.evaluate_budget_feasibility(
            budget, text_content, financial_context_scores
        )
        financial_score = financial_analysis["financial_score"]

        feature_values = {
            "novelty_score": novelty_score,
            "financial_score": financial_score,
            "budget": float(budget if budget else 0.0),
            "tech_alignment": tech_alignment,
        }
        feature_vector = np.array([[feature_values[f] for f in FEATURE_ORDER]])

        prediction_id = int(self.model.predict(feature_vector)[0])
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence_score = float(probabilities[prediction_id]) * 100

       
        if confidence_score < MIN_CONFIDENCE_FOR_AUTO_DECISION:
            final_decision = "FLAGGED FOR HUMAN REVIEW"
        else:
            final_decision = (
                "RECOMMENDED FOR APPROVAL" if prediction_id == 1 else "FLAGGED FOR HUMAN REVIEW"
            )

        ai_narrative = GeminiExplainer.generate_narrative(
            proposal_title=proposal_title,
            metrics={"novelty_score": novelty_score, "financial_score": financial_score},
            extracted_keywords=extracted_themes,
        )
        
        shap_attribution = self.shap_explainer.explain(feature_vector, prediction_id)


        final_score = (
            novelty_score * NOVELTY_WEIGHT
            + financial_score * FINANCIAL_WEIGHT
            + tech_alignment * TECHNICAL_WEIGHT
        )

        return {
            "status": "Success",
            "metrics": {
                "novelty_score": round(novelty_score, 2),
                "novelty_rating": novelty_rating,
                "financial_score": round(financial_score, 2),
                "financial_rating": financial_analysis["financial_rating"],
                "financial_risk_flags": financial_analysis["risk_flags"],
                "technical_alignment": round(tech_alignment, 2),
                "final_score": round(final_score, 2),
            },
            "explainable_ai_attribution": {
                "methodology_type": "SHAP (TreeExplainer) — per-proposal feature attribution",
                "feature_contributions": shap_attribution,
            },
            "ai_generated_narrative": ai_narrative,
            "similar_past_projects": matched_projects,
            "evaluation_verdict": {
                "decision": final_decision,
                "confidence_percentage": round(confidence_score, 2),
            },
        }