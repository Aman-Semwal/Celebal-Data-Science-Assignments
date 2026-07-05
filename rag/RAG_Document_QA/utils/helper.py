"""
helper.py - Utility functions used across the project

This file contains small helper functions that are used by
multiple parts of the application. Things like file handling,
formatting, and other common operations go here.
"""

import os
import time
import hashlib


def save_uploaded_file(uploaded_file, save_folder="data"):
    """
    Save an uploaded file from Streamlit to the data folder.
    
    Streamlit gives us a file-like object when someone uploads a file.
    This function saves it to disk so we can process it later.
    
    Args:
        uploaded_file: the file object from Streamlit's file_uploader
        save_folder: which folder to save in (default is 'data')
        
    Returns:
        The full path to the saved file
    """
    
    # Create the save folder if it doesn't exist yet
    os.makedirs(save_folder, exist_ok=True)
    
    # Build the full path where we'll save the file
    file_path = os.path.join(save_folder, uploaded_file.name)
    
    # Write the file to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def get_file_hash(file_path):
    """
    Generate a hash of a file to check if it's changed.
    
    We use this to see if a document has already been processed.
    If the hash is the same, we can skip reprocessing and just
    load the saved FAISS index.
    
    Args:
        file_path: path to the file
        
    Returns:
        A hex string hash of the file contents
    """
    
    # Read the file in binary mode and compute MD5 hash
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    
    return hasher.hexdigest()


def get_index_save_path(file_name):
    """
    Get the path where a FAISS index should be saved for a given file.
    
    Each document gets its own folder for the FAISS index, named
    after the file. This way multiple documents can each have
    their own index.
    
    Args:
        file_name: name of the PDF file (like "report.pdf")
        
    Returns:
        The folder path for storing the FAISS index
    """
    
    # Remove the file extension and use it as the folder name
    base_name = os.path.splitext(file_name)[0]
    index_path = os.path.join("data", f"faiss_index_{base_name}")
    
    return index_path


def format_time(seconds):
    """
    Format a time duration into a readable string.
    
    Args:
        seconds: number of seconds (can be a float)
        
    Returns:
        A formatted string like "1.23 seconds" or "45.6 ms"
    """
    
    if seconds < 1:
        # Show in milliseconds if less than a second
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} seconds"


def measure_time(func, *args, **kwargs):
    """
    Run a function and measure how long it takes.
    
    This is a simple wrapper that times function execution.
    We use it to show the user how long retrieval and
    generation steps take.
    
    Args:
        func: the function to run
        *args: positional arguments to pass to the function
        **kwargs: keyword arguments to pass to the function
        
    Returns:
        A tuple of (function_result, elapsed_time_in_seconds)
    """
    
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    
    return result, elapsed
