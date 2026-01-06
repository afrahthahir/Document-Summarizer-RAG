
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

class DocumentEngine:
    def __init__(self, raw_documents):
        # 1. Data Preparation: Cleaning and Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = text_splitter.split_documents(raw_documents)
        
        # 2. Document Search: Hybrid Mechanism
        # Traditional BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 3
        
        # Semantic Vector Search 
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Hybrid Ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5]
        )
        
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def get_relevant_docs(self, query):
        """Return top N relevant documents """
        return self.ensemble_retriever.invoke(query)

    def summarize(self, docs, length_instruction="succinct"):
        """Summarize retrieved documents with adjustable length"""
        prompt_template = f"""
        Write a {length_instruction} summary of the following context. 
        Capture the essence and remain coherent:
        {{text}}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain.run(docs)