"""BM25 keyword index over document chunks."""


class BM25Store:
    def __init__(self, chunks):
        raise NotImplementedError("Implement BM25Okapi per hackathon plan.")

    def search(self, query: str, k=10):
        raise NotImplementedError
