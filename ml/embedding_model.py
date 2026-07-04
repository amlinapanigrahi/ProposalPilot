from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(self, text: str):
        """Generates a dense numerical vector representation of the string text."""
        if not text.strip():
            return self.model.encode(["Empty text segment"])[0]
        return self.model.encode([text])[0]