"""FAISS index: load/split documents, build store, similarity search."""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(docs_path="memory/sample_notes"):
    raise NotImplementedError("Implement TextLoader/PyPDFLoader per hackathon plan.")


def split_documents(documents):
    raise NotImplementedError("Implement RecursiveCharacterTextSplitter per plan.")


def build_vector_store(chunks):
    raise NotImplementedError("Implement FAISS.from_documents per plan.")


def faiss_search(query: str, vector_store, k=10):
    raise NotImplementedError("Implement similarity_search per plan.")
