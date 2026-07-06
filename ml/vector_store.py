import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalVectorStore:
    def __init__(self, csv_path: str = "data/past_projects.csv"):
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.past_titles = []
        self.load_history()

    def load_history(self):
        """Loads historical projects from the CSV file to use as the comparison baseline."""
        if not os.path.exists(self.csv_path) or os.stat(self.csv_path).st_size == 0:
            return
            
        try:
            df = pd.read_csv(self.csv_path)
            if "project" in df.columns:
                self.past_titles = df["project"].fillna("").tolist()
            elif "title" in df.columns:
                self.past_titles = df["title"].fillna("").tolist()
        except Exception:
            self.past_titles = []

    def compute_novelty_score(self, target_text: str) -> float:
        """Calculates a novelty metric from 0 to 100 using TF-IDF token matching."""
        if not self.past_titles:
            return 100.0

        documents = self.past_titles + [target_text]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        
        novelty_score = (1.0 - max_similarity) * 100.0
        return round(max(0.0, min(100.0, novelty_score)), 2)

    def extract_top_keywords(self, target_text: str, top_n: int = 5) -> list:
        """
        Extracts the highest-scoring TF-IDF words from the target text
        to profile the core themes of the proposal.
        """
        try:
            tfidf_matrix = self.vectorizer.transform([target_text])
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            
            sorted_output_indices = np.argsort(tfidf_matrix.toarray()[0])[::-1]
            top_words = feature_names[sorted_output_indices[:top_n]].tolist()
            return top_words
        except Exception:
            return ["analysis", "proposal", "research", "system", "framework"]