from rouge_score import rouge_scorer
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        # RAGAS Faithfulness (LLM-based grounding check)
        client = AsyncOpenAI()
        llm = llm_factory("gpt-4o-mini", client=client)
        self.faithfulness_metric = Faithfulness(llm=llm)

    def generate_test_query(self, doc_text, llm):
        """Creates a query that ideally returns this document"""
        prompt = f"""
        Generate ONE short search query (max 12 words).
        Do NOT explain.
        Do NOT use full sentences.
        Do NOT include quotes or formatting.

        TEXT:
        {doc_text[:400]}

        QUERY:
        """
        return llm.predict(prompt).strip()

    def evaluate_summary(self, generated_summary, reference_text):
        """Measure quality using automated metrics"""
        scores = self.scorer.score(reference_text, generated_summary)
        return scores
    
    async def evaluate_faithfulness(
        self,
        user_input: str,
        response: str,
        retrieved_docs
    ):
        """
        retrieved_docs: List[Document] OR List[str]
        Evaluate faithfulness using RAGAS metric.
        The Faithfulness metric measures how factually consistent a response is with the retrieved context.
        It ranges from 0 to 1, with higher scores indicating better consistency.
        """

        # Normalize retrieved context
        if hasattr(retrieved_docs[0], "page_content"):
            contexts = [d.page_content for d in retrieved_docs]
        else:
            contexts = retrieved_docs

        result = await self.faithfulness_metric.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=contexts
        )

        return result.value
    


def interpret_rouge_performance(r1_score, rl_score):
    """
    Logic to evaluate RAG performance based on ROUGE thresholds.
    R1: Measures Keyword Capture (Unigrams)
    RL: Measures Summarization/Paraphrasing (Longest Common Subsequence)
    """
    
    # 1. Define Performance Categories
    summary_report = ""
    
    # Evaluate Keyword Capture (ROUGE-1)
    if r1_score > 0.35:
        keyword_status = "High Keyword Capture"
    elif 0.20 <= r1_score <= 0.35:
        keyword_status = "Moderate Keyword Capture"
    else:
        keyword_status = "Poor Keyword Capture"

    # Evaluate Paraphrasing/Structure (ROUGE-L)
    if rl_score > 0.20:
        paraphrase_status = "Strong Paraphrasing Power"
    elif 0.10 <= rl_score <= 0.20:
        paraphrase_status = "Low Paraphrasing Power (Extractive)"
    else:
        paraphrase_status = "Poor Structural Coherence"

    # 2. Overall Performance Logic
    if r1_score > 0.30 and rl_score > 0.15:
        overall_performance = "GOOD PERFORMANCE"
        insight = "The system successfully retrieves relevant documents and synthesizes them into accurate summaries."
    elif r1_score > 0.20:
        overall_performance = "AVERAGE PERFORMANCE"
        insight = "The system finds the right content but may struggle with coherent summarization."
    else:
        overall_performance = "POOR PERFORMANCE"
        insight = "The retrieval mechanism or the LLM prompt requires significant tuning."

    # Print the final report
    print(f"--- Performance Report ---")
    print(f"Status: {overall_performance}")
    print(f"Detail: {keyword_status} and {paraphrase_status}.")
    print(f"Justification: {insight}")