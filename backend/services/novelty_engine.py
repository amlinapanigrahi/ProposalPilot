from ml.vector_store import LocalVectorStore

class NoveltyEngine:
    def __init__(self):
        self.store = LocalVectorStore()

    def evaluate_proposal_novelty(self, text_content: str) -> dict:
        """
        Analyzes the textual content of a proposal and returns 
        a raw score along with a descriptive categorical rating.
        """
        score = self.store.compute_novelty_score(text_content)
        
        if score >= 75:
            rating = "Highly Novel"
        elif score >= 45:
            rating = "Moderately Novel"
        else:
            rating = "Low Novelty / Potential Redundancy"
            
        return {
            "novelty_score": round(score, 2),
            "novelty_rating": rating
        }