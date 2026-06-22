class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        results = self.vector_store.query(query, n_results=top_k)

        items = []
        if results and results["documents"]:
            for i, text in enumerate(results["documents"][0]):
                items.append({
                    "text": text,
                    "metadata": results["metadatas"][0][i]
                })
        return items