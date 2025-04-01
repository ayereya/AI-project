import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import uuid

# Updated LangChain imports
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate


# Local imports
from utils import (
    extract_text_from_pdf, 
    extract_metadata_from_pdf,
    extract_keywords,
    extract_entities,
    split_text_into_chunks,
    generate_embeddings,
    initialize_faiss_index,
    add_document_to_index,
    search_similar_documents,
    save_faiss_index,
    load_faiss_index,
    prepare_langchain_documents
)

# Initialize session state for storing app state
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = initialize_faiss_index()
    
if 'document_map' not in st.session_state:
    st.session_state.document_map = {}
    
if 'document_counter' not in st.session_state:
    st.session_state.document_counter = 0
    
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
    
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def init_llm():
    """Initialize the language model for question answering."""
    try:
        # Define available models
        llm_options = {
            "Flan-T5-Base (Fast but simple)": "google/flan-t5-base", 
            "Flan-T5-Large (Better quality, slower)": "google/flan-t5-large",
            "TinyLlama (Small LLM)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Phi-2 (Microsoft small model)": "microsoft/phi-2",
            "RWKV-4 (Fast inference)": "RWKV/rwkv-4-430m-pile"
        }
        
        # Let user select model in sidebar
        selected_model = st.sidebar.selectbox(
            "Select LLM for Question Answering:",
            options=list(llm_options.keys()),
            index=0,
            help="Choose a language model for answering questions. Larger models provide better quality but run slower."
        )
        
        model_name = llm_options[selected_model]
        st.sidebar.info(f"Using {selected_model} model for question answering")
        
        # Check if we should use a text-generation or text2text-generation pipeline
        if "flan-t5" in model_name.lower():
            # For T5 models (text2text-generation)
            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=512
            )
        else:
            # For decoder-only models (text-generation)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512
            )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
            
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        st.warning("Using simple context retrieval without advanced QA capabilities.")
        # Return a very simple "model" that just returns text from the context
        return None

def setup_qa_chain(texts: List[str], metadatas: List[Dict[str, Any]] = None):
    """
    Set up the question answering chain with the document store.
    
    Args:
        texts: List of document texts
        metadatas: List of metadata dictionaries
    """
    try:
        # Initialize embedding model for LangChain
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create documents for LangChain
        documents = prepare_langchain_documents(texts, metadatas)
        
        # Create vector store
        docsearch = LangchainFAISS.from_documents(documents, embeddings)
        
        # Initialize language model
        llm = init_llm()
        
        if llm:
            # Create QA chain with improved prompt template
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template="""You are an AI-powered legal analysis assistant.
                        
                        Use the following pieces of context to answer the user's question. 
                        If you don't know the answer, just say you don't know. Don't try to make up an answer.
                        
                        Context: {context}
                        
                        Question: {question}
                        
                        Answer:""",
                        input_variables=["context", "question"]
                    )
                }
            )
            
            return qa_chain
        else:
            st.warning("Using simple context retrieval without advanced QA capabilities.")
            
            # Create a simple "chain" that just returns relevant text
            class SimpleRetriever:
                def __init__(self, docsearch):
                    self.docsearch = docsearch
                
                def invoke(self, query):
                    docs = self.docsearch.similarity_search(query, k=3)
                    return {
                        "result": "\n\n".join([doc.page_content for doc in docs]),
                        "source_documents": docs
                    }
                
                # Legacy support for run method
                def run(self, query):
                    return self.invoke(query)
            
            return SimpleRetriever(docsearch)
            
    except Exception as e:
        st.error(f"Error setting up QA chain: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """
    Process an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Extract text and metadata
        text = extract_text_from_pdf(temp_file_path)
        metadata = extract_metadata_from_pdf(temp_file_path)
        
        # Extract keywords and entities
        keywords = extract_keywords(text)
        entities = extract_entities(text)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        # Add chunks to FAISS index
        chunk_ids = list(range(
            st.session_state.document_counter, 
            st.session_state.document_counter + len(chunks)
        ))
        
        st.session_state.faiss_index, doc_map = add_document_to_index(
            st.session_state.faiss_index, 
            chunks, 
            chunk_ids
        )
        
        # Update document map
        st.session_state.document_map.update(doc_map)
        
        # Update document counter
        st.session_state.document_counter += len(chunks)
        
        # Prepare chunk metadata for each chunk
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadatas.append({
                "filename": uploaded_file.name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                **{k: v for k, v in metadata.items() if k in ["Title", "Author", "Creation Date"]}
            })
        
        # Set up QA chain if not already set up
        if st.session_state.qa_chain is None:
            st.session_state.qa_chain = setup_qa_chain(chunks, chunk_metadatas)
        else:
            # Update the QA chain with new documents
            st.session_state.qa_chain = setup_qa_chain(
                chunks + [st.session_state.document_map[i] for i in st.session_state.document_map],
                chunk_metadatas
            )
        
        # Add to uploaded docs
        doc_id = str(uuid.uuid4())
        st.session_state.uploaded_docs.append({
            "id": doc_id,
            "name": uploaded_file.name,
            "metadata": metadata,
            "keywords": keywords,
            "entities": entities,
            "chunks": chunks,
            "chunk_ids": chunk_ids
        })
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return doc_id
        
    except Exception as e:
        # Clean up the temporary file
        os.unlink(temp_file_path)
        raise e

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="AI-Powered eDiscovery & Legal Analytics",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ AI-Powered eDiscovery & Legal Analytics")
    st.markdown("""
    This application helps legal professionals analyze documents using AI techniques including:
    - Document metadata extraction
    - Keyword analysis
    - Named Entity Recognition (NER)
    - Document embedding and similarity search
    - AI-powered question answering
    """)
    
    # Sidebar
    st.sidebar.title("Document Operations")
    
    # Settings section
    with st.sidebar.expander("Application Settings", expanded=False):
        # Memory management settings
        max_chunks = st.slider(
            "Max Chunks per Document", 
            min_value=10, 
            max_value=100, 
            value=30,
            help="Higher values allow processing larger documents but use more memory"
        )
        
        chunk_size = st.slider(
            "Chunk Size (characters)", 
            min_value=500, 
            max_value=2000, 
            value=1000,
            help="Size of text chunks for processing. Smaller chunks may give more precise results"
        )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Process the file
        try:
            with st.spinner("Processing document..."):
                doc_id = process_uploaded_file(uploaded_file)
                st.sidebar.success(f"Document '{uploaded_file.name}' processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "🔍 Search & QA", "📊 Document Statistics"])
    
    with tab1:
        if st.session_state.uploaded_docs:
            # Select document
            doc_options = {doc["name"]: doc["id"] for doc in st.session_state.uploaded_docs}
            selected_doc_name = st.selectbox("Select a document to analyze:", list(doc_options.keys()))
            selected_doc_id = doc_options[selected_doc_name]
            
            # Get the selected document
            selected_doc = next((doc for doc in st.session_state.uploaded_docs if doc["id"] == selected_doc_id), None)
            
            if selected_doc:
                # Document metadata
                st.subheader("Document Metadata")
                # Fix for Arrow serialization error - ensure all values are strings
                metadata_df = pd.DataFrame({
                    "Metadata": list(selected_doc["metadata"].keys()), 
                    "Value": [str(v) for v in selected_doc["metadata"].values()]
                })
                st.dataframe(metadata_df, use_container_width=True)
                
                # Keywords
                st.subheader("Top Keywords")
                keywords_cols = st.columns(3)
                for i, col in enumerate(keywords_cols):
                    chunk_size = len(selected_doc["keywords"]) // 3
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < 2 else len(selected_doc["keywords"])
                    col.write(", ".join(selected_doc["keywords"][start_idx:end_idx]))
                
                # Named Entities
                st.subheader("Named Entities")
                
                entity_cols = st.columns(len(selected_doc["entities"]) if selected_doc["entities"] else 1)
                
                if selected_doc["entities"]:
                    for i, (entity_type, entities) in enumerate(selected_doc["entities"].items()):
                        with entity_cols[i]:
                            st.write(f"**{entity_type}**")
                            st.write(", ".join(entities[:10]))
                            if len(entities) > 10:
                                with st.expander(f"Show all {len(entities)} {entity_type}"):
                                    st.write(", ".join(entities))
                else:
                    st.write("No named entities found in this document.")
                
                # Document preview
                st.subheader("Document Preview")
                preview_text = " ".join(selected_doc["chunks"][:2])  # Show first two chunks
                st.text_area("Preview", preview_text, height=200)
        else:
            st.info("Please upload a document to begin analysis.")
    
    with tab2:
        if st.session_state.uploaded_docs:
            st.subheader("Search & Question Answering")
            
            search_query = st.text_input("Enter your question or search query:")
            
            search_button = st.button("Search")
            
            if search_query and search_button:
                with st.spinner("Searching..."):
                    # Perform QA if chain is available
                    if st.session_state.qa_chain:
                        try:
                            # Use invoke() instead of run() to avoid deprecation warning
                            if hasattr(st.session_state.qa_chain, 'invoke'):
                                result = st.session_state.qa_chain.invoke(search_query)
                            else:
                                result = st.session_state.qa_chain.run(search_query)
                            
                            # Display answer
                            st.subheader("Answer")
                            
                            if isinstance(result, dict) and "result" in result:
                                # Handle the case for SimpleRetriever
                                st.write(result["result"])
                                source_docs = result.get("source_documents", [])
                            else:
                                # Handle the case for LangChain RetrievalQA
                                st.write(result)
                                source_docs = []  # No source documents in simple string result
                                
                            # Show source documents if available
                            if source_docs:
                                with st.expander("View Source Passages"):
                                    for i, doc in enumerate(source_docs):
                                        st.markdown(f"**Source {i+1}**")
                                        st.write(doc.page_content)
                                        st.write("---")
                            
                        except Exception as e:
                            st.error(f"Error in question answering: {e}")
                    
                    # Also perform similarity search
                    try:
                        similar_docs = search_similar_documents(
                            st.session_state.faiss_index,
                            search_query,
                            st.session_state.document_map,
                            k=3
                        )
                        
                        st.subheader("Similar Passages")
                        for i, (doc_id, doc_text, score) in enumerate(similar_docs):
                            with st.expander(f"Passage {i+1} (Similarity: {score:.2f})"):
                                st.write(doc_text)
                    
                    except Exception as e:
                        st.error(f"Error in similarity search: {e}")
        else:
            st.info("Please upload a document to enable search and question answering.")
    
    with tab3:
        if st.session_state.uploaded_docs:
            st.subheader("Document Statistics")
            
            # Document count
            st.metric("Total Documents", len(st.session_state.uploaded_docs))
            
            # Document chunks
            total_chunks = sum(len(doc["chunks"]) for doc in st.session_state.uploaded_docs)
            st.metric("Total Text Chunks", total_chunks)
            
            # Document sizes
            doc_sizes = []
            for doc in st.session_state.uploaded_docs:
                doc_text = " ".join(doc["chunks"])
                doc_sizes.append((doc["name"], len(doc_text)))
            
            # Display document sizes
            st.subheader("Document Sizes (characters)")
            size_cols = st.columns(len(doc_sizes) if doc_sizes else 1)
            
            for i, (doc_name, doc_size) in enumerate(doc_sizes):
                with size_cols[i]:
                    st.metric(doc_name, f"{doc_size:,}")
        else:
            st.info("Please upload a document to view statistics.")

if __name__ == "__main__":
    main()