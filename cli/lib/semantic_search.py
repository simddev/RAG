import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.cache_path = "cache/movie_embeddings.npy"

    def generate_embedding(self, text: str):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or whitespace.")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents

        self.document_map = {}
        texts = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            texts.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        os.makedirs("cache", exist_ok=True)
        np.save(self.cache_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.cache_path):
            cached = np.load(self.cache_path)

            if len(cached) == len(documents):
                self.embeddings = cached
                return self.embeddings

        return self.build_embeddings(documents)


# -------------------------
# TOP LEVEL FUNCTIONS
# -------------------------


def embed_query_text(query: str) -> None:
    search = SemanticSearch()

    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_model() -> None:
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search = SemanticSearch()

    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
