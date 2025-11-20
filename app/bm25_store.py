from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, docs):
        self.texts = [d.page_content for d in docs]
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]
