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
]


EQUIPMENT_EXEMPLARS = [
    "The requested funds will procure a GPU compute cluster, laboratory "
    "instrumentation, and sensor hardware, itemized in the attached budget "
    "breakdown with per-unit costs and vendor quotes.",
    "Capital expenditure covers specialized microscopy equipment and a "
    "dedicated server rack, with itemized costs detailed by category.",
    "The budget allocates funds toward acquiring a robotic testbed, "
    "calibration instruments, and associated maintenance contracts.",
]

TIMELINE_EXEMPLARS = [
    "The project spans 18 months, structured into three sequential "
    "six-month phases: design, implementation, and validation.",
    "Work will be completed over a 24-month period, with quarterly "
    "milestones and a mid-project review at month 12.",
    "The proposed timeline covers two years, beginning with a three-month "
    "planning phase followed by iterative development sprints.",
]

SCALABILITY_EXEMPLARS = [
    "A six-month pilot deployment with 200 users will precede full-scale "
    "rollout, allowing iterative refinement before broader adoption.",
    "The system will first be validated at small scale in a single "
    "facility, with a defined scalability plan for expansion to "
    "additional sites pending pilot results.",
    "We propose a phased scale-up strategy: initial proof-of-concept, "
    "followed by a limited pilot, followed by enterprise-wide deployment.",
]


class LocalVectorStore:
    def __init__(self, csv_path: str = "data/past_projects.csv"):
        self.csv_path = csv_path

        self.novelty_vectorizer = TfidfVectorizer(stop_words="english")
        self.technical_vectorizer = TfidfVectorizer(stop_words="english")
        self.equipment_vectorizer = TfidfVectorizer(stop_words="english")
        self.timeline_vectorizer = TfidfVectorizer(stop_words="english")
        self.scalability_vectorizer = TfidfVectorizer(stop_words="english")

        self.corpus_titles = []      # for display in similar_past_projects
        self.corpus_documents = []   # title + abstract, used for similarity
        self.novelty_matrix = None   # fitted once, reused across requests

        self.load_history()
        self._fit_technical_corpus()
        self._fit_financial_corpora()

    def load_history(self):
        """Loads past projects (title + abstract) as the novelty comparison baseline."""
        if not os.path.exists(self.csv_path) or os.stat(self.csv_path).st_size == 0:
            print(f"No past_projects.csv, novelty scoring will default to 100.")
            return

        try:
            df = pd.read_csv(self.csv_path)

            if "title" in df.columns:
                titles = df["title"].fillna("")
            else:
                titles = pd.Series([""] * len(df))

            if "abstract" in df.columns:
                abstract = df["abstract"].fillna("")
            else:
                abstract = pd.Series([""] * len(df))

            self.corpus_titles = titles.tolist()
            self.corpus_documents = (titles + ". " + abstract).tolist()

            if self.corpus_documents:
                self.novelty_matrix = self.novelty_vectorizer.fit_transform(self.corpus_documents)
                print(f"Loaded {len(self.corpus_documents)} past projects from {self.csv_path}.")

        except Exception as e:
            print(f"Failed to load {self.csv_path}: {e}")
            self.corpus_titles = []
            self.corpus_documents = []
            self.novelty_matrix = None

    def _fit_technical_corpus(self):
        """Fits a separate vectorizer against curated strong-technical-writing exemplars."""
        self.technical_matrix = self.technical_vectorizer.fit_transform(TECHNICAL_EXEMPLARS)

    def _fit_financial_corpora(self):
        """Fits separate vectorizers for each financial feasibility sub-check."""
        self.equipment_matrix = self.equipment_vectorizer.fit_transform(EQUIPMENT_EXEMPLARS)
        self.timeline_matrix = self.timeline_vectorizer.fit_transform(TIMELINE_EXEMPLARS)
        self.scalability_matrix = self.scalability_vectorizer.fit_transform(SCALABILITY_EXEMPLARS)

    def analyze(self, target_text: str, top_n_similar: int = 3, top_n_keywords: int = 5) -> dict:
        """Runs novelty/technical/similarity/keyword analysis in one pass.
        Transforms target_text once per vectorizer, reuses the resulting
        vector across every derived output."""
        novelty_vec = (
            self.novelty_vectorizer.transform([target_text])
            if self.novelty_matrix is not None else None
        )
        technical_vec = self.technical_vectorizer.transform([target_text])

        return {
            "novelty_score": self._novelty_from_vec(novelty_vec),
            "technical_alignment": self._corpus_similarity_from_vec(technical_vec, self.technical_matrix),
            "similar_projects": self._top_similar_from_vec(novelty_vec, top_n_similar),
            "top_keywords": self._keywords_from_vec(novelty_vec, top_n_keywords),
        }

    def compute_financial_context_scores(self, target_text: str) -> dict:
        """Returns 0-100 similarity scores for equipment, timeline, and
        scalability dimensions, based on similarity to curated exemplars."""
        return {
            "equipment_justification_score": self._corpus_similarity(
                target_text, self.equipment_vectorizer, self.equipment_matrix
            ),
            "timeline_clarity_score": self._corpus_similarity(
                target_text, self.timeline_vectorizer, self.timeline_matrix
            ),
            "scalability_score": self._corpus_similarity(
                target_text, self.scalability_vectorizer, self.scalability_matrix
            ),
        }

    def _corpus_similarity(self, target_text: str, vectorizer, matrix) -> float:
        """Transforms target_text, then scores it against a reference corpus.
        Use when you don't already have a transformed vector."""
        target_vec = vectorizer.transform([target_text])
        return self._corpus_similarity_from_vec(target_vec, matrix)

    def _corpus_similarity_from_vec(self, target_vec, matrix) -> float:
        """Mean of the top-k cosine similarities against a reference corpus,
        given an ALREADY-TRANSFORMED vector. Shared by technical alignment
        (called from analyze(), which pre-transforms) and all three
        financial dimensions (called via _corpus_similarity above)."""
        similarities = cosine_similarity(target_vec, matrix).flatten()
        k = min(3, len(similarities))
        top_k = np.sort(similarities)[-k:] if k > 0 else similarities
        score = float(np.mean(top_k)) * 100.0 if len(top_k) > 0 else 0.0
        return round(max(0.0, min(100.0, score)), 2)

    def _novelty_from_vec(self, target_vec) -> float:
        """0-100: higher = less similar to any past proposal (more novel)."""
        if target_vec is None or not self.corpus_documents:
            return 100.0

        similarities = cosine_similarity(target_vec, self.novelty_matrix).flatten()
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0

        novelty_score = (1.0 - max_similarity) * 100.0
        return round(max(0.0, min(100.0, novelty_score)), 2)

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