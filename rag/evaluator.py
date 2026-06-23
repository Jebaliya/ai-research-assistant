from rouge_score import rouge_scorer


class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def evaluate(self, question, answer, retrieved_chunks):
        # Context se text nikalo
        context = " ".join([chunk["text"] for chunk in retrieved_chunks])

        # ROUGE score calculate karo
        scores = self.scorer.score(context, answer)
        rouge1 = round(scores['rouge1'].fmeasure * 100, 2)
        rougeL = round(scores['rougeL'].fmeasure * 100, 2)

        # Relevance score
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        common = question_words & answer_words
        relevance = round(len(common) / max(len(question_words), 1) * 100, 2)

        # Context coverage
        context_words = set(context.lower().split())
        answer_word_set = set(answer.lower().split())
        coverage = round(len(answer_word_set & context_words) / max(len(answer_word_set), 1) * 100, 2)

        return {
            "relevance": relevance,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "context_coverage": coverage
        }