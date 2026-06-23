class ResearchAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def run(self, question):
        steps = []

        # Step 1 — Query Analysis
        steps.append({
            "step": "🔍 Query Analysis",
            "result": f"Analyzing question: '{question}'"
        })

        # Step 2 — Document Retrieval
        results = self.retriever.retrieve(question, top_k=3)
        steps.append({
            "step": "📄 Document Retrieval",
            "result": f"Found {len(results)} relevant chunks from: {set([r['metadata'].get('source', 'unknown') for r in results])}"
        })

        if not results:
            steps.append({
                "step": "❌ No Results",
                "result": "No relevant documents found."
            })
            return steps, "I could not find relevant information in the uploaded documents."

        # Step 3 — Context Building
        context = "\n\n".join([r["text"] for r in results])
        steps.append({
            "step": "🧩 Context Building",
            "result": f"Built context from {len(results)} chunks ({len(context)} chars)"
        })

        # Step 4 — Answer Generation
        steps.append({
            "step": "🤖 Answer Generation",
            "result": "Generating answer using LLM..."
        })
        answer = self.llm.generate_answer(question, results, "")

        # Step 5 — Done
        steps.append({
            "step": "✅ Complete",
            "result": "Answer generated successfully"
        })

        return steps, answer