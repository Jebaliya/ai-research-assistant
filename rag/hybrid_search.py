from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    return re.findall(r'\w+', text.lower())


class HybridSearch:
    def __init__(self):
        self.documents = []
        self.bm25 = None

    def add_documents(self, documents):
        self.documents.extend(documents)
        tokenized = [tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):
        if not self.documents or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score > 0:
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "bm25_score": score
                })
        return results