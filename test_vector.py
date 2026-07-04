# test_vector.py
from ml.vector_store import LocalVectorStore

store = LocalVectorStore()
# Test sample with content slightly resembling the grid optimization entry
sample_abstract = "We intend to coordinate electrical microgrid clusters via smart machine algorithms."
score = store.compute_novelty_score(sample_abstract)

print(f"Computed Project Novelty Rating: {score:.2f} / 100")