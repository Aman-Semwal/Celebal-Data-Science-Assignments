"""
text_splitter.py - Breaks long text into smaller chunks

When we have a big document, we can't just send the whole thing
to the AI model. We need to break it into smaller pieces (chunks)
so that we can find the most relevant parts for a question.

This module uses LangChain's RecursiveCharacterTextSplitter which
is a smart way to split text - it tries to keep paragraphs and
sentences together instead of cutting in the middle of a word.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text_into_chunks(text, chunk_size=800, chunk_overlap=150):
    """
    Split a large text into smaller overlapping chunks.
    
    The chunks overlap a little bit so that if an important piece
    of information is at the boundary between two chunks, it won't
    be lost. Think of it like a sliding window over the text.
    
    Args:
        text: the full text string to split up
        chunk_size: maximum number of characters per chunk (default 800)
        chunk_overlap: how many characters to overlap between chunks (default 150)
        
    Returns:
        A dictionary with:
            - 'chunks': list of text chunks
            - 'num_chunks': how many chunks were created
            - 'avg_chunk_size': average size of each chunk in characters
    """
    
    # Create the splitter with our settings
    # The separators list tells it to try splitting by paragraphs first,
    # then by newlines, then by sentences, then by words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # use simple character count for length
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the text into chunks
    chunks = splitter.split_text(text)
    
    # Calculate some basic stats about the chunks
    num_chunks = len(chunks)
    
    # Calculate average chunk size
    # Need to handle the edge case where there are no chunks
    if num_chunks > 0:
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = round(total_chars / num_chunks, 1)
    else:
        avg_chunk_size = 0
    
    # Return everything in a nice dictionary
    result = {
        "chunks": chunks,
        "num_chunks": num_chunks,
        "avg_chunk_size": avg_chunk_size
    }
    
    return result
