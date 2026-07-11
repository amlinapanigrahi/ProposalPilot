import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reference snippets representing strong technical writing:
TECHNICAL_EXEMPLARS = [
    "We propose a multi-stage pipeline combining convolutional feature extraction "
    "with a transformer-based sequence model, validated via k-fold cross-validation "
    "against a held-out test set with quantified precision and recall metrics.",

    "The proposed synthesis route employs a two-step catalytic process optimized "
    "through response surface methodology, with reaction kinetics characterized "
    "via mass spectrometry and validated against control samples.",

    "Our approach formulates the resource allocation problem as a mixed-integer "
    "linear program, solved using a branch-and-bound algorithm with proven "
    "convergence bounds under specified constraint sets.",

    "The system architecture employs a distributed consensus protocol with "
    "Byzantine fault tolerance, benchmarked for throughput and latency under "
    "varying network partition scenarios.",

    "We design a randomized controlled trial with stratified sampling across "
    "three cohorts, using a mixed-effects regression model to control for "
    "confounding variables and establish statistical significance.",

    "The proposed sensor fusion framework combines Kalman filtering with a "
    "learned correction model, evaluated against ground-truth trajectories "
    "using root-mean-square error as the primary metric.",

    "Our methodology applies finite element analysis to model structural stress "
    "distribution under cyclic loading, with results validated against "
    "physical prototype testing at three load thresholds.",

    "We implement a differentially private aggregation scheme with formally "
    "bounded epsilon-delta guarantees, benchmarked against standard "
    "federated learning baselines on held-out data.",
]


class LocalVectorStore:
    def __init__(self, csv_path: str = "data/past_projects.csv"):
        self.csv_path = csv_path

        self.novelty_vectorizer = TfidfVectorizer(stop_words="english")
        self.technical_vectorizer = TfidfVectorizer(stop_words="english")

        self.corpus_titles = []      # for display in similar_past_projects
        self.corpus_documents = []   # title + abstract, used for similarity
        self.novelty_matrix = None   # fitted once, reused across requests

        self.load_history()
        self._fit_technical_corpus()

    def load_history(self):
        """Loads past projects (title + abstract) as the novelty comparison baseline."""
        if not os.path.exists(self.csv_path) or os.stat(self.csv_path).st_size == 0:
            print(f"[LocalVectorStore] No past_projects.csv found at {self.csv_path}, "
                  f"novelty scoring will default to 100.")
            return

        try:
            df = pd.read_csv(self.csv_path)

            if "title" in df.columns:
                titles = df["title"].fillna("")
            else:
                titles = pd.Series([""] * len(df))

            if "abstracts" in df.columns:
                abstracts = df["abstracts"].fillna("")
            else:
                abstracts = pd.Series([""] * len(df))
            

            self.corpus_titles = titles.tolist()
            self.corpus_documents = (titles + ". " + abstracts).tolist()

            if self.corpus_documents:
                self.novelty_matrix = self.novelty_vectorizer.fit_transform(self.corpus_documents)

        except Exception as e:
            print("Failed to load")
            self.corpus_titles = []
            self.corpus_documents = []
            self.novelty_matrix = None

    def _fit_technical_corpus(self):
        """Fits a separate vectorizer against curated strong-technical-writing exemplars."""
        self.technical_matrix = self.technical_vectorizer.fit_transform(TECHNICAL_EXEMPLARS)

    def analyze(self, target_text: str, top_n_similar: int = 3, top_n_keywords: int = 5) -> dict:

        novelty_vec = (
            self.novelty_vectorizer.transform([target_text])
            if self.novelty_matrix is not None else None
        )

        technical_vec = self.technical_vectorizer.transform([target_text])

        return {
            "novelty_score": self._novelty_from_vec(novelty_vec),
            "technical_alignment": self._technical_from_vec(technical_vec),
            "similar_projects": self._top_similar_from_vec(novelty_vec, top_n_similar),
            "top_keywords": self._keywords_from_vec(novelty_vec, top_n_keywords),
        }

    
    def _novelty_from_vec(self, target_vec) -> float:
        """0-100: higher = less similar to any past proposal (more novel)."""
        if target_vec is None or not self.corpus_documents:
            return 100.0

        similarities = cosine_similarity(target_vec, self.novelty_matrix).flatten()
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0

        novelty_score = (1.0 - max_similarity) * 100.0
        return round(max(0.0, min(100.0, novelty_score)), 2)

    def _technical_from_vec(self, target_vec) -> float:
        """0-100: higher = more similar to strong technical-writing exemplars."""
        similarities = cosine_similarity(target_vec, self.technical_matrix).flatten()

        # Mean of top-3 exemplar
        top_k = np.sort(similarities)[-3:] if len(similarities) >= 3 else similarities
        alignment_score = float(np.mean(top_k)) * 100.0
        return round(max(0.0, min(100.0, alignment_score)), 2)

    def _top_similar_from_vec(self, target_vec, top_n: int) -> list:
        """Most similar past proposals with similarity scores.
        Powers `similar_past_projects` in the API response."""
        if target_vec is None or not self.corpus_documents:
            return []

        similarities = cosine_similarity(target_vec, self.novelty_matrix).flatten()
        top_n = min(top_n, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_n]

        return [
            {
                "title": self.corpus_titles[i],
                "similarity_score": round(float(similarities[i]) * 100.0, 2),
            }
            for i in top_indices
        ]

    def _keywords_from_vec(self, target_vec, top_n: int) -> list:
        """Highest-scoring TF-IDF words from the target text."""
        if target_vec is None:
            return []

        try:
            feature_names = np.array(self.novelty_vectorizer.get_feature_names_out())
            sorted_indices = np.argsort(target_vec.toarray()[0])[::-1]
            top_words = feature_names[sorted_indices[:top_n]].tolist()
            return top_words
        except Exception as e:
            print(f"[LocalVectorStore] Keyword extraction failed: {e}")
            return []