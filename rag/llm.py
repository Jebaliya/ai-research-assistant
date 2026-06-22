import os
from groq import Groq


class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)

    def generate_answer(self, question, retrieved_chunks, chat_history_text):
        context = "\n\n".join([
            f"Source: {item['metadata'].get('source', 'unknown')}\n{item['text']}"
            for item in retrieved_chunks
        ])

        prompt = (
            "You are a helpful AI research assistant.\n\n"
            "Conversation so far:\n"
            + chat_history_text +
            "\n\nContext from uploaded documents:\n"
            + context +
            "\n\nUser question: " + question +
            "\n\nRules:"
            "\n1. Answer the question directly. Do NOT start with 'In the context of uploaded documents' or mention source at beginning."
            "\n2. Give a clean direct answer."
            "\n3. At the very end, on a new line write exactly: Source: [filename]"
            "\n4. If answer not found say: I could not find this in the uploaded documents."
        )

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content