"""
pdf_loader.py - Handles reading and extracting text from PDF files

This module takes a PDF file path and pulls out all the text
from every page. It uses PyPDF2 which is a simple library for
working with PDF documents.
"""

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path):
    """
    Read a PDF file and extract text from all its pages.
    
    Takes the path to a PDF file, opens it, and goes through
    each page to get the text content. If a page has no text
    (like a scanned image), it just skips that page.
    
    Args:
        file_path: string path to the PDF file on disk
        
    Returns:
        A dictionary with:
            - 'full_text': all the text combined into one string
            - 'num_pages': how many pages the PDF has
            - 'total_chars': total number of characters extracted
            - 'page_texts': list of text from each individual page
    """
    
    # Open the PDF file and create a reader object
    reader = PdfReader(file_path)
    
    # Get the total number of pages
    num_pages = len(reader.pages)
    
    # We'll collect text from each page in this list
    page_texts = []
    
    # Loop through every page and extract its text
    for page_number in range(num_pages):
        page = reader.pages[page_number]
        
        # Try to extract text from this page
        text = page.extract_text()
        
        # Sometimes a page might return None or empty string
        # In that case we just add an empty string
        if text is None:
            text = ""
        
        # Clean up extra whitespace
        text = text.strip()
        
        page_texts.append(text)
    
    # Combine all pages into one big text string
    # Using newlines between pages so they don't blend together
    full_text = "\n\n".join(page_texts)
    
    # Count total characters
    total_chars = len(full_text)
    
    # Put everything in a dictionary and return it
    result = {
        "full_text": full_text,
        "num_pages": num_pages,
        "total_chars": total_chars,
        "page_texts": page_texts
    }
    
    return result
