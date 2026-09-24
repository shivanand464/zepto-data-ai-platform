from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------

def load_documents():
    """Load all policy documents from the docs directory."""

    documents = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


# ---------------------------------------------------------
# Create embeddings and ChromaDB collection
# ---------------------------------------------------------

def build_vector_database():
    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 policy documents, but found {len(documents)}."
        )

    print(f"Loaded {len(documents)} policy documents.")

    # Load the required local embedding model.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [document["text"] for document in documents]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Persistent ChromaDB client.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate the collection so that every ingestion run starts
    # from the current policy corpus.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[document["id"] for document in documents],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": document["source"]}
            for document in documents
        ],
    )

    print("\nChromaDB ingestion complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored documents: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


# ---------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------

def test_retrieval():
    """Run a simple retrieval test against ChromaDB."""

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query = "How long does Zepto delivery take?"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query: {query}\n")

    for index, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][index - 1]
        distance = results["distances"][0][index - 1]

        print(f"Result {index}")
        print(f"Source: {metadata['source']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {document[:200]}...")
        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_database()
    test_retrieval()