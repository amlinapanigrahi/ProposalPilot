import os
import pandas as pd
import numpy as np
from ml.embedding_model import EmbeddingEngine

class LocalVectorStore:
    def __init__(self, csv_path: str = "data/past_projects.csv"):
        self.csv_path = csv_path
        self.encoder = EmbeddingEngine()
        self.past_embeddings = []
        self.past_metadata = []
        self.load_and_index_history()

    def load_and_index_history(self):
        """Reads the CSV file data source and caches its vector footprints."""
        if not os.path.exists(self.csv_path):
            return
            
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            text_to_embed = f"{row['title']}. {row['abstract']}"
            vector = self.encoder.get_embeddings(text_to_embed)
            
            self.past_embeddings.append(vector)
            self.past_metadata.append({
                "id": row["id"],
                "title": row["title"],
                "approved": row["approved"]
            })
            
        if self.past_embeddings:
            self.past_embeddings = np.array(self.past_embeddings)

    def compute_novelty_score(self, target_text: str) -> float:
        """
        Calculates a novelty metric from 0 to 100.
        100 means completely unique; 0 means a direct carbon-copy duplicate.
        """
        if len(self.past_embeddings) == 0:
            return 100.0

        target_vector = self.encoder.get_embeddings(target_text)
        
        dot_products = np.dot(self.past_embeddings, target_vector)
        norm_history = np.linalg.norm(self.past_embeddings, axis=1)
        norm_target = np.linalg.norm(target_vector)
        
        similarities = dot_products / (norm_history * norm_target + 1e-8)
        max_similarity = float(np.max(similarities))
        
        novelty_score = (1.0 - max_similarity) * 100.0
        return max(0.0, min(100.0, novelty_score))