from rag.hybrid_search import HybridSearch


class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.hybrid = HybridSearch()

    def add_to_hybrid(self, documents):
        self.hybrid.add_documents(documents)

    def retrieve(self, query, top_k=5):
        # Vector search
        vector_results = self.vector_store.query(query, n_results=top_k)
        vector_items = []
        if vector_results and vector_results["documents"]:
            for i, text in enumerate(vector_results["documents"][0]):
                vector_items.append({
                    "text": text,
                    "metadata": vector_results["metadatas"][0][i],
                    "score": 1.0
                })

        # BM25 search
        bm25_items = self.hybrid.search(query, top_k=top_k)

        # Combine
        combined = {}

        for item in vector_items:
            key = item["text"][:100]
            combined[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "score": item["score"]
            }

        for item in bm25_items:
            key = item["text"][:100]
            if key in combined:
                combined[key]["score"] += item["bm25_score"] * 0.3
            else:
                combined[key] = {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": item["bm25_score"] * 0.3
                }

        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        return sorted_results