"""FastAPI app for the target RAG chatbot. Built in M1.

Endpoints:
- POST /chat   {message, session_id} -> {answer, retrieved_doc_ids}
- POST /ingest {doc_id, text}        -> adds a document (how poisoned docs get in)
- POST /reset                        -> restores the clean knowledge base
- GET  /health

Defenses are switched on per request with the `X-RedRAG-Defenses` header, so a
single scan can compare "no defenses" against "all defenses" without restarting.
"""
