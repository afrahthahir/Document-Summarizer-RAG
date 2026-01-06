from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def generate_test_query(self, doc_text, llm):
        """Creates a query that ideally returns this document"""
        response = llm.predict(f"Generate a specific search query that would find this text: {doc_text[:500]}")
        return response

    def evaluate_summary(self, generated_summary, reference_text):
        """Measure quality using automated metrics"""
        scores = self.scorer.score(reference_text, generated_summary)
        return scores
    

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

# Use your actual scores
# interpret_rouge_performance(0.3956, 0.2271)
