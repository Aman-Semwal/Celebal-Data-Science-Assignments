"""
rag_pipeline.py - The main RAG (Retrieval-Augmented Generation) pipeline

This is where everything comes together. This module:
1. Takes the user's question
2. Finds relevant chunks from the document (retrieval)
3. Builds a prompt with the context
4. Sends it to Google Gemini to get an answer (generation)

The key idea of RAG is that we don't just ask the AI a question
directly - we first find the relevant information from our document
and include it in the prompt. This way the AI can give accurate
answers based on the actual document content.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv


# Load the API key from the .env file
load_dotenv()


# This is our prompt template
# It tells the AI to only answer using the provided context
PROMPT_TEMPLATE = """You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:
'I could not find the answer in the uploaded document.'

Context:
{context}

Question:
{question}

Answer:"""


def setup_gemini():
    """
    Initialize the Gemini API with the API key from .env file.
    
    You need to have GOOGLE_API_KEY set in your .env file.
    Get a free API key from: https://aistudio.google.com/app/apikey
    
    Returns:
        A Gemini GenerativeModel object ready to generate text
        
    Raises:
        ValueError: if the API key is not found in environment
    """
    
    # Load the API key from the .env file, overriding any existing env vars
    from dotenv import load_dotenv
    import os
    
    # Explicitly find the .env file in the project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Get the API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Make sure the API key exists
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "Please set your GOOGLE_API_KEY in the .env file!\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    
    # Configure the Gemini API with our key
    genai.configure(api_key=api_key)
    
    # Create the model - using the canonical model path for the SDK
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    return model


def build_prompt(context_chunks, question):
    """
    Build the prompt that we'll send to Gemini.
    
    Takes the retrieved context chunks and the user's question,
    and formats them into our prompt template.
    
    Args:
        context_chunks: list of text strings from the document
        question: the user's question
        
    Returns:
        The formatted prompt string
    """
    
    # Join all the context chunks together with some separation
    # We number each chunk so the AI can reference specific parts
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"[Chunk {i}]\n{chunk}")
    
    # Combine all chunks with blank lines between them
    combined_context = "\n\n".join(context_parts)
    
    # Fill in the prompt template
    prompt = PROMPT_TEMPLATE.format(
        context=combined_context,
        question=question
    )
    
    return prompt


def generate_answer(model, prompt):
    """
    Send the prompt to Gemini and get an answer.
    
    Args:
        model: the Gemini GenerativeModel
        prompt: the formatted prompt string
        
    Returns:
        The generated answer text from Gemini
    """
    
    # Send the prompt to Gemini and get a response
    response = model.generate_content(prompt)
    
    # Extract just the text from the response
    answer = response.text
    
    return answer


def ask_question(vector_store, question, num_chunks=4):
    """
    The main function that ties the whole RAG pipeline together.
    
    This is the function that app.py calls when the user asks
    a question. It handles the full flow from question to answer.
    
    Args:
        vector_store: the FAISS vector store with our document
        question: what the user wants to know
        num_chunks: how many context chunks to retrieve (default 4)
        
    Returns:
        A dictionary with:
            - 'answer': the generated answer text
            - 'context_chunks': the retrieved chunks used
            - 'similarity_scores': how similar each chunk was
    """
    
    # Import the search function from our embedding store module
    from modules.embedding_store import search_similar_chunks
    
    # Step 1: Find the most relevant chunks from the document
    # The vector store handles converting the question to an embedding
    # and finding the closest matches
    search_results = search_similar_chunks(
        vector_store, question, num_results=num_chunks
    )
    
    # Separate the chunks and their similarity scores
    context_chunks = []
    similarity_scores = []
    for doc, score in search_results:
        context_chunks.append(doc.page_content)
        similarity_scores.append(round(float(score), 4))
    
    # Step 2: Set up Gemini
    model = setup_gemini()
    
    # Step 3: Build the prompt with context + question
    prompt = build_prompt(context_chunks, question)
    
    # Step 4: Get the answer from Gemini
    answer = generate_answer(model, prompt)
    
    # Step 5: Package everything up and return it
    result = {
        "answer": answer,
        "context_chunks": context_chunks,
        "similarity_scores": similarity_scores
    }
    
    return result
