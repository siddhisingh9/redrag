"""The RAG pipeline: chunk -> embed -> store -> retrieve -> build prompt. Built in M1.

- Chunking splits each handbook document into overlapping pieces, so a
  retrieved piece is small enough to fit the prompt but still makes sense.
- Embeddings come from `BAAI/bge-small-en-v1.5` on CPU (leaves the 4 GB GPU for Ollama).
- Chunks are stored in a local, persistent ChromaDB collection.
- The top-k most similar chunks are pasted into the prompt as context.
"""
