"""Simple local vector store for Ollama embeddings without OpenSearch dependency."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger


class SimpleVectorStore:
    """Lightweight in-memory vector store using cosine similarity."""

    def __init__(self, index_name: str = "medical_guidelines", storage_dir: str = "data/vector_store"):
        self._logger = get_logger("audra.services.simple_vector_store")
        self.index_name = index_name
        self.storage_path = Path(storage_dir) / f"{index_name}.pkl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        # Load existing index if available
        if self.storage_path.exists():
            self._load_index()
            self._logger.info(f"Loaded existing index with {len(self.documents)} documents.")
        else:
            self._logger.info("Initialized empty vector store.")

    def index_document(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Index a single document."""
        doc = {
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }

        # Check if doc_id already exists
        existing_idx = next((i for i, d in enumerate(self.documents) if d["id"] == doc_id), None)

        if existing_idx is not None:
            # Update existing document
            self.documents[existing_idx] = doc
            if self.embeddings is not None:
                self.embeddings[existing_idx] = np.array(embedding)
        else:
            # Add new document
            self.documents.append(doc)
            if self.embeddings is None:
                self.embeddings = np.array([embedding])
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])

        self._logger.debug(f"Indexed document: {doc_id}")

    def index_batch(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Bulk index documents."""
        for doc in documents:
            self.index_document(
                doc_id=doc["id"],
                text=doc["text"],
                embedding=doc["embedding"],
                metadata=doc.get("metadata", {})
            )
        self._save_index()
        self._logger.info(f"Indexed {len(documents)} documents in batch.")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            self._logger.warning("No documents in vector store.")
            return []

        # Convert query to numpy array and normalize
        query_vec = np.array(query_embedding)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Normalize stored embeddings
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)

        # Apply filters if provided
        valid_indices = list(range(len(self.documents)))
        if filters:
            valid_indices = [
                i for i in valid_indices
                if self._matches_filters(self.documents[i]["metadata"], filters)
            ]

        if not valid_indices:
            return []

        # Get top-k indices from valid documents
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in valid_similarities[:top_k]]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(similarities[idx])
            })

        return results

    def delete_index(self) -> None:
        """Delete all documents and the stored index."""
        self.documents = []
        self.embeddings = None
        if self.storage_path.exists():
            self.storage_path.unlink()
        self._logger.info("Deleted vector store index.")

    def get_document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self.documents)

    def ping(self) -> bool:
        """Health check - always returns True for in-memory store."""
        return True

    def _save_index(self) -> None:
        """Save index to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        with open(self.storage_path, "wb") as f:
            pickle.dump(data, f)
        self._logger.debug(f"Saved index to {self.storage_path}")

    def _load_index(self) -> None:
        """Load index from disk."""
        with open(self.storage_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = np.array(data["embeddings"]) if data["embeddings"] else None

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
