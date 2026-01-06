from rag_engine import DocumentEngine
from evaluator import Evaluator, interpret_rouge_performance
from langchain_community.document_loaders import TextLoader

# 1. Setup the RAG System and Evaluator
loader = TextLoader("temp.txt")
engine = DocumentEngine(loader.load())
eval_tool = Evaluator()

# 2. Automated Test Case (As required by the assignment)
# Step A: Pick a document chunk to test
test_doc = engine.chunks[0].page_content 

# Step B: Generate a synthetic query for that chunk
test_query = eval_tool.generate_test_query(test_doc, engine.llm)
print(f"Generated Query: {test_query}")

# Step C: Run the RAG Pipeline
retrieved_docs = engine.get_relevant_docs(test_query)
generated_summary = engine.summarize(retrieved_docs, length_instruction="succinct")

# Step D: Evaluate Results
# We compare the generated summary against the original source text
scores = eval_tool.evaluate_summary(generated_summary, test_doc)

print("\n--- Evaluation Results ---")
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# Step E: Interpret Performance
interpret_rouge_performance(scores['rouge1'].fmeasure, scores['rougeL'].fmeasure)