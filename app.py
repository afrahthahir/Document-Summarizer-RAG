import streamlit as st
from langchain_community.document_loaders import TextLoader
from rag_engine import DocumentEngine

st.set_page_config(page_title="Document RAG System")
st.title("Document Search & Summarization")

# Sidebar for Summary Settings
summary_len = st.sidebar.select_slider(
    "Select Summary Length",
    options=["One-sentence", "Succinct", "Detailed Paragraph"]
)

# File Upload 
uploaded_file = st.file_uploader("Upload Corpus (TXT)")

if uploaded_file:
    # Simulating data ingestion and cleaning 
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = TextLoader("temp.txt")
    engine = DocumentEngine(loader.load())

    query = st.text_input("Enter your query:") 
    
    if query:
        results = engine.get_relevant_docs(query)
        
        st.subheader("Summary")
        summary = engine.summarize(results, length_instruction=summary_len)
        st.write(summary) 

        st.divider()
        st.subheader("Source Excerpts (Pagination)") 
        # Basic pagination implementation
        for idx, doc in enumerate(results):
            with st.expander(f"Source Document {idx+1}"):
                st.write(doc.page_content)