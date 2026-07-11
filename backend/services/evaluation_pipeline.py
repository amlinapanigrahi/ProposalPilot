import os
import pickle
import numpy as np
from backend.services.financial_checker import FinancialChecker
from backend.services.document_parser import extract_text_from_pdf
from ml.vector_store import LocalVectorStore
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

    def evaluate_incoming_proposal(
        self, text_content: str, budget: float, proposal_title: str = "Untitled Proposal"
    ) -> dict:
        """
        Runs the proposal through all analysis layers and outputs a final
        predictive administrative recommendation.
        """
        # One call, transforms text_content once per vectorizer internally —
        # replaces separate compute_novelty_score / compute_technical_alignment /
        # get_top_similar / extract_top_keywords calls that each re-transformed
        # the same text.
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

        financial_analysis = FinancialChecker.evaluate_budget_feasibility(budget, text_content)
        financial_score = financial_analysis["financial_score"]

        # FEATURE_ORDER from config.py fixes the column order in one place —
        # must match the order used in train_model.py's training matrix.
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

        # Confidence-threshold override: don't trust a borderline prediction
        # just because it crossed 50%. Below the threshold, defer to a human
        # regardless of which class the model predicted.
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

        # Gini/MDI importance is a GLOBAL measure — fixed at training time,
        # identical for every proposal regardless of its own content. It
        # answers "which features did the model rely on overall," not
        # "why did THIS proposal get THIS prediction" (that would be SHAP,
        # a per-instance explanation method not currently implemented).
        importances = self.model.feature_importances_
        importance_by_feature = dict(zip(FEATURE_ORDER, importances))

        # Real multi-factor score, replacing the old novelty-only final_score.
        # Uses the same weights as the synthetic training labels in
        # train_model.py, so the score and the classifier's ground truth
        # reflect the same definition of proposal quality.
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
                "technical_alignment": round(tech_alignment, 2),
                "final_score": round(final_score, 2),
            },
            "explainable_ai_attribution": {
                "novelty_impact_weight": round(importance_by_feature["novelty_score"] * 100, 1),
                "financial_feasibility_weight": round(importance_by_feature["financial_score"] * 100, 1),
                "technical_density_weight": round(importance_by_feature["tech_alignment"] * 100, 1),
                "budget_weight": round(importance_by_feature["budget"] * 100, 1),
                "methodology_type": "Gini Importance / Mean Decrease in Impurity (global, not per-proposal)",
            },
            "ai_generated_narrative": ai_narrative,
            "similar_past_projects": matched_projects,
            "evaluation_verdict": {
                "decision": final_decision,
                "confidence_percentage": round(confidence_score, 2),
            },
        }