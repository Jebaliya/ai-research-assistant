import chromadb
import os


class ChromaVectorStore:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="research_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents):
        ids = [doc["id"] for doc in documents]
        existing = self.collection.get(ids=ids)["ids"]
        new_docs = [doc for doc in documents if doc["id"] not in existing]

        if not new_docs:
            return

        self.collection.add(
            ids=[doc["id"] for doc in new_docs],
            documents=[doc["text"] for doc in new_docs],
            metadatas=[doc["metadata"] for doc in new_docs]
        )

    def query(self, query_text, n_results=5):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results