from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------

def load_documents():
    """Load all policy documents from the docs directory."""

    documents = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(
            encoding="utf-8",
        ).strip()

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
# Chunk documents
# ---------------------------------------------------------

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    """
    Split a document into overlapping text chunks.

    The overlap helps preserve context between neighboring
    chunks during retrieval.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def create_chunks(documents):
    """
    Create retrieval chunks while preserving the original
    document filename as source metadata.
    """

    chunks = []

    for document in documents:
        document_chunks = chunk_text(document["text"])

        for chunk_index, chunk in enumerate(
            document_chunks,
            start=1,
        ):
            chunks.append(
                {
                    "id": (
                        f"{document['id']}"
                        f"_chunk_{chunk_index}"
                    ),
                    "text": chunk,
                    "source": document["source"],
                    "chunk_index": chunk_index,
                }
            )

    return chunks


# ---------------------------------------------------------
# Create embeddings and ChromaDB collection
# ---------------------------------------------------------

def build_vector_database():
    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 policy documents, "
            f"but found {len(documents)}."
        )

    print(f"Loaded {len(documents)} policy documents.")

    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the policy documents."
        )

    print(f"Created {len(chunks)} text chunks.")
    print(
        f"Chunk size: {CHUNK_SIZE} characters | "
        f"Overlap: {CHUNK_OVERLAP} characters"
    )

    # Load the required local embedding model.
    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Persistent ChromaDB client.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Recreate the collection so every ingestion run
    # starts from the current policy corpus.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print(
            f"Deleted existing collection: "
            f"{COLLECTION_NAME}"
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[
            chunk["id"]
            for chunk in chunks
        ],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    print("\nChromaDB ingestion complete.")
    print(
        f"Collection: {COLLECTION_NAME}"
    )
    print(
        f"Stored chunks: {collection.count()}"
    )
    print(
        f"Source documents: {len(documents)}"
    )
    print(
        f"Database location: {CHROMA_DIR}"
    )


# ---------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------

def test_retrieval():
    """Run a simple retrieval test against ChromaDB."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    query = "How long does Zepto delivery take?"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query: {query}\n")

    for index, document in enumerate(
        results["documents"][0],
        start=1,
    ):
        metadata = results["metadatas"][0][index - 1]
        distance = results["distances"][0][index - 1]

        print(f"Result {index}")
        print(
            f"Source: {metadata['source']}"
        )
        print(
            f"Chunk: {metadata['chunk_index']}"
        )
        print(
            f"Cosine distance: {distance:.4f}"
        )
        print(
            f"Text: {document[:200]}..."
        )
        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_database()
    test_retrieval()