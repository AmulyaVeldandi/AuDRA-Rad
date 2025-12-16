"""Index medical guidelines using Ollama embeddings and simple vector store."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.guidelines.indexer import GuidelineIndexer
from src.services.ollama_embeddings import OllamaEmbeddingClient
from src.services.simple_vector_store import SimpleVectorStore
from src.utils.logger import get_logger

logger = get_logger("index_guidelines_ollama")


def main():
    """Index guideline documents using Ollama embeddings."""
    guidelines_dir = project_root / "data" / "guidelines"

    if not guidelines_dir.exists():
        logger.error(f"Guidelines directory not found: {guidelines_dir}")
        return 1

    # Initialize Ollama embedding client
    logger.info("Initializing Ollama embedding client...")
    try:
        embedding_client = OllamaEmbeddingClient(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
    except Exception as exc:
        logger.error(f"Failed to initialize Ollama embedding client: {exc}")
        return 1

    # Initialize simple vector store
    logger.info("Initializing vector store...")
    vector_store = SimpleVectorStore(index_name="medical_guidelines")

    # Delete existing index
    logger.info("Clearing existing index...")
    vector_store.delete_index()

    # Initialize guideline indexer
    logger.info("Initializing guideline indexer...")
    indexer = GuidelineIndexer()

    # Load and parse guidelines
    logger.info(f"Loading guidelines from {guidelines_dir}...")
    markdown_files = list(guidelines_dir.glob("*.md"))
    logger.info(f"Found {len(markdown_files)} guideline files")

    total_chunks = 0
    for md_file in markdown_files:
        logger.info(f"Processing {md_file.name}...")
        try:
            chunks = indexer.load_and_chunk(md_file)
            logger.info(f"  Generated {len(chunks)} chunks")

            # Create embeddings and index
            documents = []
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = embedding_client.get_document_embedding(chunk.text)

                    # Prepare document
                    doc = {
                        "id": f"{md_file.stem}_chunk_{i}",
                        "text": chunk.text,
                        "embedding": embedding,
                        "metadata": {
                            "source": chunk.source,
                            "category": chunk.category,
                            "size_min_mm": chunk.size_min_mm,
                            "size_max_mm": chunk.size_max_mm,
                            "risk_level": chunk.risk_level,
                            "modality": chunk.modality,
                            "recommendation": chunk.recommendation,
                        }
                    }
                    documents.append(doc)

                except Exception as exc:
                    logger.warning(f"  Failed to process chunk {i}: {exc}")
                    continue

            # Batch index documents
            if documents:
                vector_store.index_batch(documents)
                total_chunks += len(documents)
                logger.info(f"  Indexed {len(documents)} chunks from {md_file.name}")

        except Exception as exc:
            logger.error(f"Failed to process {md_file.name}: {exc}")
            continue

    logger.info(f"✓ Indexing complete! Total chunks indexed: {total_chunks}")
    logger.info(f"✓ Vector store has {vector_store.get_document_count()} documents")

    return 0


if __name__ == "__main__":
    sys.exit(main())
