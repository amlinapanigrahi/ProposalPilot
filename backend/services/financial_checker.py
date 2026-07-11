class FinancialChecker:
    
    EQUIPMENT_THRESHOLD = 45.0
    TIMELINE_THRESHOLD = 45.0
    SCALABILITY_THRESHOLD = 45.0

    MAX_EQUIPMENT_PENALTY = 20.0
    MAX_TIMELINE_PENALTY = 15.0
    MAX_SCALABILITY_PENALTY = 15.0

    EQUIPMENT_CHECK_BUDGET_FLOOR = 1_000_000    # > 10 lakh INR
    SCALABILITY_CHECK_BUDGET_FLOOR = 5_000_000  # > 50 lakh INR

    @staticmethod
    def _proportional_penalty(score: float, threshold: float, max_penalty: float) -> float:
        """Scales the penalty by how far below threshold the score falls"""

        if score >= threshold:
            return 0.0
        
        gap_ratio = (threshold - score) / threshold
        
        return round(max_penalty * gap_ratio, 2)

    @staticmethod
    def evaluate_budget_feasibility(
        budget: float, proposal_text: str, financial_context_scores: dict) -> dict:
        
        if budget is None or budget <= 0:
            return {
                "financial_score": 40.0,
                "financial_rating": "Unverifiable (Missing Budget Inputs)",
                "risk_flags": ["No explicit budget metrics declared."],
            }

        equipment_score = financial_context_scores["equipment_justification_score"]
        timeline_score = financial_context_scores["timeline_clarity_score"]
        scalability_score = financial_context_scores["scalability_score"]

        risk_flags = []
        base_score = 100.0

        if budget > FinancialChecker.EQUIPMENT_CHECK_BUDGET_FLOOR:
            penalty = FinancialChecker._proportional_penalty(
                equipment_score,
                FinancialChecker.EQUIPMENT_THRESHOLD,
                FinancialChecker.MAX_EQUIPMENT_PENALTY,
            )
            if penalty > 0:
                base_score -= penalty
                risk_flags.append(
                    f"High capital request with weak equipment/infrastructure "
                    f"justification (similarity score: {equipment_score:.1f}/100)."
                )

        timeline_penalty = FinancialChecker._proportional_penalty(
            timeline_score,
            FinancialChecker.TIMELINE_THRESHOLD,
            FinancialChecker.MAX_TIMELINE_PENALTY,
        )
        if timeline_penalty > 0:
            base_score -= timeline_penalty
            risk_flags.append(
                f"Project timeline/duration not clearly articulated "
                f"(similarity score: {timeline_score:.1f}/100)."
            )

        if budget > FinancialChecker.SCALABILITY_CHECK_BUDGET_FLOOR:
            scalability_penalty = FinancialChecker._proportional_penalty(
                scalability_score,
                FinancialChecker.SCALABILITY_THRESHOLD,
                FinancialChecker.MAX_SCALABILITY_PENALTY,
            )
            if scalability_penalty > 0:
                base_score -= scalability_penalty
                risk_flags.append(
                    f"Enterprise-scale funding request lacking clear pilot/"
                    f"scalability strategy (similarity score: {scalability_score:.1f}/100)."
                )

        if base_score >= 80:
            rating = "Highly Feasible"
        elif base_score >= 50:
            rating = "Moderately Feasible"
        else:
            rating = "High Financial Risk Profile"

        return {
            "financial_score": max(0.0, round(base_score, 2)),
            "financial_rating": rating,
            "risk_flags": risk_flags,
        }