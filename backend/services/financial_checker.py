# backend/services/financial_checker.py

class FinancialChecker:
    @staticmethod
    def evaluate_budget_feasibility(budget: float, proposal_text: str) -> dict:
        """
        Assesses the realism of the funding request against context clues 
        found inside the proposal text documents.
        """
        if budget is None or budget <= 0:
            return {
                "financial_score": 40.0,
                "financial_rating": "Unverifiable (Missing Budget Inputs)",
                "risk_flags": ["No explicit budget metrics declared."]
            }
            
        lower_text = proposal_text.lower()
        risk_flags = []
        base_score = 100.0
        
        if "equipment" not in lower_text and "infrastructure" not in lower_text:
            if budget > 1000000:  # If budget is > 10 Lakhs but no hardware mentioned
                base_score -= 20
                risk_flags.append("High capital request missing matching equipment/infrastructure itemization.")
                
        # Heuristic 2: Check for duration matching context
        if "year" not in lower_text and "month" not in lower_text:
            base_score -= 15
            risk_flags.append("Project execution timeline/duration boundaries not clearly declared in text.")
            
        # Heuristic 3: Check for disproportionately high prototyping costs
        if budget > 5000000:  # > 50 Lakhs
            if "scalability" not in lower_text and "pilot" not in lower_text:
                base_score -= 15
                risk_flags.append("Enterprise scale funding request lacking pilot execution or scalability strategies.")
                
        # Classify the adjusted score bounds
        if base_score >= 80:
            rating = "Highly Feasible"
        elif base_score >= 50:
            rating = "Moderately Feasible"
        else:
            rating = "High Financial Risk Profile"
            
        return {
            "financial_score": max(0.0, float(base_score)),
            "financial_rating": rating,
            "risk_flags": risk_flags
        }