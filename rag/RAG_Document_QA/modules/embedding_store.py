"""
embedding_store.py - Creates embeddings and manages the FAISS vector store

This module does two important things:
1. Converts text chunks into numerical vectors (embeddings) using
   a HuggingFace model that runs locally on your computer
2. Stores those vectors in a FAISS index so we can quickly find
   the most similar chunks when someone asks a question

FAISS (Facebook AI Similarity Search) is a library that makes
searching through lots of vectors really fast.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# This is the embedding model we'll use
# It's small (about 80MB) and works pretty well for general text
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Load the HuggingFace embedding model.
    
    This model converts text into a 384-dimensional vector.
    It runs locally so we don't need any API keys for this part.
    
    Returns:
        A HuggingFaceEmbeddings object ready to use
    """
    
    # Create the embedding model
    # model_kwargs sets it to use CPU (change to 'cuda' if you have a GPU)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embeddings


def create_vector_store(text_chunks, embeddings_model):
    """
    Create a FAISS vector store from text chunks.
    
    This takes our text chunks, converts them to embeddings using
    the model, and puts them into a FAISS index. The index lets
    us quickly find which chunks are most similar to a question.
    
    Args:
        text_chunks: list of strings (our document chunks)
        embeddings_model: the HuggingFace embedding model to use
        
    Returns:
        A FAISS vector store object
    """
    
    # FAISS.from_texts does all the heavy lifting:
    # 1. Converts each chunk to an embedding vector
    # 2. Adds all vectors to the FAISS index
    # 3. Keeps track of which chunk goes with which vector
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings_model
    )
    
    return vector_store


def save_vector_store(vector_store, save_path):
    """
    Save the FAISS index to disk so we don't have to rebuild it.
    
    This is useful because generating embeddings can take a while,
    especially for large documents. By saving the index, we can
    just load it next time instead of reprocessing everything.
    
    Args:
        vector_store: the FAISS vector store to save
        save_path: folder path where the index files will be stored
    """
    
    # Make sure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the FAISS index and the associated texts
    vector_store.save_local(save_path)


def load_vector_store(save_path, embeddings_model):
    """
    Load a previously saved FAISS index from disk.
    
    Args:
        save_path: folder path where the index files are stored
        embeddings_model: the same embedding model that was used to create it
        
    Returns:
        A FAISS vector store object, or None if the file doesn't exist
    """
    
    # Check if the saved index exists
    index_file = os.path.join(save_path, "index.faiss")
    if not os.path.exists(index_file):
        return None
    
    # Load the FAISS index
    # allow_dangerous_deserialization=True is needed because FAISS
    # uses pickle to save/load, and LangChain wants us to confirm
    # that we trust the source of the file
    vector_store = FAISS.load_local(
        save_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    
    return vector_store


def search_similar_chunks(vector_store, query, num_results=4):
    """
    Find the most similar chunks to a given query.
    
    This is the retrieval step - when someone asks a question,
    we convert the question to an embedding and find which chunks
    in our document are closest to it in vector space.
    
    Args:
        vector_store: the FAISS vector store to search
        query: the question text from the user
        num_results: how many similar chunks to return (default 4)
        
    Returns:
        A list of tuples, each containing (document, score)
        where document has the chunk text and score is the
        similarity score (lower is more similar in FAISS)
    """
    
    # similarity_search_with_score returns both the documents
    # and how similar they are to the query
    results = vector_store.similarity_search_with_score(
        query,
        k=num_results
    )
    
    return results
