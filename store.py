"""
store.py
========

Vector store and retrieval module for "The Unofficial Guide" RAG project.

This script:
    1. Initializes a persistent ChromaDB client (data saved to ./chroma_db).
    2. Loads a local sentence-transformer embedding model
       (all-MiniLM-L6-v2).
    3. Provides `add_chunks_to_db(chunks)` to embed and store chunk
       dictionaries (as produced by ingest.py) in the "housing_reviews"
       collection, with `source_file` and `chunk_index` stored as metadata.
    4. Provides `retrieve(query, k=4)` to embed a query and return the
       top-k most similar chunks with their text, metadata, and distance.

Run directly (`python3 store.py`) to test retrieval with a sample query.
"""

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_db"          # Local folder where ChromaDB persists data
COLLECTION_NAME = "housing_reviews"  # Name of the collection holding our chunks
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

# A PersistentClient writes its index/data to disk at CHROMA_PATH, so the
# collection survives across script runs (no need to re-embed every time).
_client = chromadb.PersistentClient(path=CHROMA_PATH)

# get_or_create_collection() either fetches the existing "housing_reviews"
# collection (if it was created in a previous run) or creates a fresh one.
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

# Load the embedding model once at module import time so it isn't reloaded
# on every call to add_chunks_to_db() or retrieve().
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Embedding Pipeline
# ---------------------------------------------------------------------------

def add_chunks_to_db(chunks):
    """
    Embed a list of chunk dictionaries and store them in the ChromaDB
    "housing_reviews" collection.

    Each `chunk` is expected to be a dict with at least:
        - "text":        the chunk's text content
        - "source_file": the originating file's name
        - "chunk_index": the integer position of the chunk within its
                          source document

    How ChromaDB stores metadata alongside vectors:
        ChromaDB's `collection.add()` takes four parallel lists -- `ids`,
        `embeddings`, `documents`, and `metadatas` -- where the i-th entry
        of each list all describe the SAME item. Internally, Chroma stores
        the embedding vector in its vector index (for similarity search)
        while the `documents` text and `metadatas` dict are stored
        alongside it in a regular record store, keyed by the same `id`.
        When you query the vector index, Chroma uses the matching ids to
        look up and return the associated document text and metadata
        together with the similarity results -- so metadata "rides along"
        with each vector without affecting the similarity computation
        itself.

    Returns the number of chunks added.
    """

    if not chunks:
        print("[INFO] No chunks provided to add_chunks_to_db(); nothing to do.")
        return 0

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        source_file = chunk["source_file"]
        chunk_index = chunk["chunk_index"]
        text = chunk["text"]

        # Build a unique, human-readable ID per chunk by combining the
        # source file name with its chunk index, e.g.
        # "discord_log_2024.txt_chunk_3". This keeps IDs stable and
        # traceable back to their origin.
        chunk_id = f"{source_file}_chunk_{chunk_index}"

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append({
            "source_file": source_file,
            "chunk_index": chunk_index,
        })

    # Generate embeddings for all chunk texts in one batch call (more
    # efficient than embedding one at a time). `convert_to_numpy=True`
    # returns a numpy array; ChromaDB also accepts plain lists, so we
    # convert to lists for broad compatibility.
    embeddings = _embedding_model.encode(documents, convert_to_numpy=True).tolist()

    # `collection.add()` upserts-by-id is NOT guaranteed across all Chroma
    # versions (older versions error on duplicate ids), so for a simple
    # ingestion script we just call add(). If you re-run ingestion on the
    # same data, consider deleting the collection first or using
    # collection.upsert() if available in your installed chromadb version.
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[INFO] Added {len(ids)} chunks to collection '{COLLECTION_NAME}'.")
    return len(ids)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query, k=4):
    """
    Embed `query` with the same embedding model used for ingestion, search
    the "housing_reviews" collection for the top-k most similar chunks, and
    return a structured list of results.

    Each result dict contains:
        - "text":        the retrieved chunk's text
        - "source_file": the source file metadata for the chunk
        - "chunk_index": the chunk's index within its source file
        - "distance":    the geometric distance score from the query
                          embedding (lower = more similar)

    Returns a list of result dicts, ordered from most to least similar.
    """

    # Embed the query the same way chunk text was embedded, so both live
    # in the same vector space and distances are meaningful.
    query_embedding = _embedding_model.encode([query], convert_to_numpy=True).tolist()

    raw_results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    results = []

    # Chroma returns results as parallel lists-of-lists (one outer list per
    # query embedding -- we only passed one query, so we use index [0]).
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        results.append({
            "text": text,
            "source_file": metadata.get("source_file"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distance,
        })

    return results


# ---------------------------------------------------------------------------
# Test Harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_query = (
        "What do students say about the commute or traffic timings "
        "from campus to the Enclave?"
    )

    print(f"Querying collection '{COLLECTION_NAME}' for:\n  \"{sample_query}\"\n")

    top_results = retrieve(sample_query, k=3)

    if not top_results:
        print("No results found. Have you run add_chunks_to_db() on your "
              "ingested chunks yet?")
    else:
        print(f"Top {len(top_results)} retrieved chunk(s):\n")
        for i, result in enumerate(top_results, start=1):
            print("-" * 70)
            print(f"RESULT {i}")
            print(f"  Source file : {result['source_file']}")
            print(f"  Chunk index : {result['chunk_index']}")
            print(f"  Distance    : {result['distance']:.4f}")
            print("  Text:")
            print("  " + "-" * 40)
            for line in result["text"].splitlines():
                print(f"  {line}")
            print("  " + "-" * 40)
        print("-" * 70)
