
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import DistanceStrategy


from dotenv import load_dotenv

load_dotenv()

class DocumentEngine:
    def __init__(self, raw_documents):
        # 1. Data Preparation: Cleaning and Chunking
        # Reduced the chunk size to 400 for finer granularity
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(raw_documents)
        
        # 2. Document Search: Hybrid Mechanism
        # Traditional BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 3
        
        # Semantic Vector Search 
        embeddings = OpenAIEmbeddings()
        # Using cosine similarity for better semantic matching
        vectorstore = FAISS.from_documents(self.chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"search_type": "similarity_score_threshold", "k": 3, "score_threshold": 0.35})
        
        # Hybrid Ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5]
        )
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_relevant_docs(self, query):
        docs = self.ensemble_retriever.invoke(query)

        # If vector retriever found nothing meaningful → reject
        vector_docs = self.vector_retriever.get_relevant_documents(query)

        if not vector_docs:
            return []

        return docs
    

    def summarize(self, docs, length_instruction="succinct"):

        if not docs:
            return "Information not found in documents."
        # Strict Prompt for Grounded Summarization without hallucinations
        prompt_template = f"""
        
        You are a strict document assistant. Summarize the context below into a {length_instruction} version.
    
        STRICT RULES:
        1. Use ONLY information from the CONTEXT block.
        2. Do NOT add outside facts, tips, or information even if you know they are true.
        3. If the context is empty, say "Information not found in documents."
        
        CONTEXT:
        {{text}}

        GROUNDED SUMMARY:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain.run(docs)