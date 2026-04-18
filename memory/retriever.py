"""RRF fusion (BM25 + FAISS) and cross-encoder reranking."""


def rrf_fusion(faiss_docs, bm25_results, k=60):
    raise NotImplementedError("Implement RRF per hackathon plan.")


def cross_encoder_rerank(query: str, docs, top_k=3):
    raise NotImplementedError("Implement CrossEncoder per hackathon plan.")


def retrieve(query: str, vector_store, bm25_store):
    raise NotImplementedError("Wire faiss_search + BM25 + RRF + rerank per plan.")
