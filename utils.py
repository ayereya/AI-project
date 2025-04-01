import fitz  # PyMuPDF
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Any
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    doc = fitz.open(file_path)
    text = ""
    
    for page in doc:
        text += page.get_text()
        
    return text

def extract_metadata_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing metadata
    """
    doc = fitz.open(file_path)
    metadata = doc.metadata
    
    # Convert to a more friendly format
    friendly_metadata = {
        "Title": metadata.get("title", "N/A"),
        "Author": metadata.get("author", "N/A"),
        "Subject": metadata.get("subject", "N/A"),
        "Keywords": metadata.get("keywords", "N/A"),
        "Creator": metadata.get("creator", "N/A"),
        "Producer": metadata.get("producer", "N/A"),
        "Creation Date": metadata.get("creationDate", "N/A"),
        "Modification Date": metadata.get("modDate", "N/A"),
        "Number of Pages": str(len(doc))  # Convert to string to avoid Arrow serialization issues
    }
    
    return friendly_metadata

def extract_keywords(text: str, limit: int = 30) -> List[str]:
    """
    Extract keywords (nouns) from text using spaCy.
    
    Args:
        text: Input text
        limit: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    doc = nlp(text[:100000])  # Limit text size to avoid memory issues
    
    # Extract nouns and proper nouns
    keywords = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "PROPN")]
    
    # Count frequency
    keyword_freq = {}
    for keyword in keywords:
        if keyword.lower() in keyword_freq:
            keyword_freq[keyword.lower()] += 1
        else:
            keyword_freq[keyword.lower()] = 1
    
    # Sort by frequency
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [keyword for keyword, _ in sorted_keywords[:limit]]

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with entity types as keys and lists of entities as values
    """
    # Process text in chunks to avoid memory issues
    max_chunk_size = 100000
    entities = {}
    
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i+max_chunk_size]
        doc = nlp(chunk)
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
    
    return entities

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better embedding and retrieval.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        NumPy array of embeddings
    """
    # Process in batches to avoid memory issues with large documents
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

def initialize_faiss_index(dimension: int = 384) -> faiss.IndexFlatL2:
    """
    Initialize a FAISS index for storing document embeddings.
    
    Args:
        dimension: Dimension of the embeddings
        
    Returns:
        FAISS index
    """
    return faiss.IndexFlatL2(dimension)

def add_document_to_index(index: faiss.IndexFlatL2, documents: List[str], 
                         document_ids: List[int]) -> Tuple[faiss.IndexFlatL2, Dict[int, str]]:
    """
    Add documents to FAISS index.
    
    Args:
        index: FAISS index
        documents: List of document texts
        document_ids: List of document IDs
        
    Returns:
        Updated FAISS index and document map
    """
    # Generate embeddings
    embeddings = generate_embeddings(documents)
    
    # Add to index
    index.add(np.array(embeddings).astype('float32'))
    
    # Create a mapping from index to document text
    document_map = {i: doc for i, doc in zip(document_ids, documents)}
    
    return index, document_map

def search_similar_documents(index: faiss.IndexFlatL2, query: str, 
                            document_map: Dict[int, str], k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Search for documents similar to the query.
    
    Args:
        index: FAISS index
        query: Query text
        document_map: Mapping from index to document text
        k: Number of similar documents to retrieve
        
    Returns:
        List of tuples (document_id, document_text, similarity_score)
    """
    # Generate query embedding
    query_embedding = model.encode([query])[0].reshape(1, -1).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k=k)
    
    # Get documents
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:  # FAISS may return -1 if there are not enough results
            continue
        results.append((idx, document_map[idx], float(distances[0][i])))
    
    return results

def save_faiss_index(index: faiss.IndexFlatL2, document_map: Dict[int, str], path: str = "faiss_index"):
    """
    Save FAISS index and document map to disk.
    
    Args:
        index: FAISS index
        document_map: Mapping from index to document text
        path: Path to save the index
    """
    os.makedirs(path, exist_ok=True)
    
    # Save the index
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    
    # Save the document map
    with open(os.path.join(path, "document_map.pkl"), "wb") as f:
        pickle.dump(document_map, f)

def load_faiss_index(path: str = "faiss_index") -> Tuple[faiss.IndexFlatL2, Dict[int, str]]:
    """
    Load FAISS index and document map from disk.
    
    Args:
        path: Path to load the index from
        
    Returns:
        FAISS index and document map
    """
    # Load the index
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    
    # Load the document map
    with open(os.path.join(path, "document_map.pkl"), "rb") as f:
        document_map = pickle.load(f)
    
    return index, document_map

def prepare_langchain_documents(texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[LangchainDocument]:
    """
    Prepare documents for LangChain.
    
    Args:
        texts: List of document texts
        metadatas: List of metadata dictionaries
        
    Returns:
        List of LangChain documents
    """
    if metadatas is None:
        metadatas = [{}] * len(texts)
    
    documents = []
    for i, (text, metadata) in enumerate(zip(texts, metadatas)):
        documents.append(
            LangchainDocument(page_content=text, metadata={"source": f"document_{i}", **metadata})
        )
    
    return documents