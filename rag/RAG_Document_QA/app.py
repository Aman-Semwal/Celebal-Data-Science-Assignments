"""
app.py - Main Streamlit application for RAG Document QA

This is the entry point of the application. It creates the web
interface using Streamlit where users can:
1. Upload PDF documents
2. Process them (extract text, create embeddings, build FAISS index)
3. Ask questions and get answers from the document

Run this with: streamlit run app.py
"""

# Disable Streamlit file watcher and fix PyTorch classes path inspection issue on Python 3.14
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

import sys
import time
import streamlit as st

# Add the project root to the path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.pdf_loader import extract_text_from_pdf
from modules.text_splitter import split_text_into_chunks
from modules.embedding_store import (
    get_embedding_model,
    create_vector_store,
    save_vector_store,
    load_vector_store,
)
from modules.rag_pipeline import ask_question
from utils.helper import (
    save_uploaded_file,
    get_file_hash,
    get_index_save_path,
    format_time,
)


# ──────────────────────────────────────────────
# Page configuration - must be the first Streamlit command
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoInsight - AI Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Cache the embedding model so it loads only once
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the HuggingFace embedding model and cache it."""
    return get_embedding_model()


# ──────────────────────────────────────────────
# Initialize session state variables
# ──────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing_log" not in st.session_state:
    st.session_state.processing_log = []


def add_log(message):
    """Add a message to the processing log."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.processing_log.append(f"[{timestamp}] {message}")


# ──────────────────────────────────────────────
# Sidebar - Document Upload & Processing
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("Upload a PDF document to get started.")
    
    # File uploader - accepts PDF files
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload one PDF file to ask questions about it.",
    )
    
    # Show file info if a file is uploaded
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    # Process Document button
    process_button = st.button(
        "⚙️ Process Document",
        use_container_width=True,
        disabled=(uploaded_file is None),
        help="Extract text, create chunks, and build the search index.",
    )
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Show document status
    st.markdown("---")
    st.subheader("📊 Document Status")
    
    if st.session_state.doc_processed:
        info = st.session_state.doc_info
        st.success("✅ Document is ready!")
        st.metric("Pages", info.get("num_pages", "—"))
        st.metric("Characters", f"{info.get('total_chars', 0):,}")
        st.metric("Chunks", info.get("num_chunks", "—"))
        st.metric("Avg Chunk Size", f"{info.get('avg_chunk_size', 0)} chars")
    else:
        st.warning("⏳ No document processed yet.")


# ──────────────────────────────────────────────
# Process the uploaded document
# ──────────────────────────────────────────────
if process_button and uploaded_file is not None:
    # Reset the processing log
    st.session_state.processing_log = []
    
    # Create a progress area in the main page
    progress_container = st.empty()
    
    with progress_container.container():
        st.subheader("🔄 Processing Document...")
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        
        try:
            # ── Step 1: Save the uploaded file ──
            status_text.text("💾 Saving uploaded file...")
            progress_bar.progress(10, text="Saving file...")
            file_path = save_uploaded_file(uploaded_file)
            add_log(f"Saved file: {uploaded_file.name}")
            
            # Check if we already have a saved index for this file
            file_hash = get_file_hash(file_path)
            index_path = get_index_save_path(uploaded_file.name)
            
            # ── Step 2: Extract text from PDF ──
            status_text.text("📖 Extracting text from PDF...")
            progress_bar.progress(20, text="Extracting text...")
            
            start = time.time()
            pdf_data = extract_text_from_pdf(file_path)
            extract_time = time.time() - start
            
            add_log(
                f"Extracted {pdf_data['total_chars']:,} chars "
                f"from {pdf_data['num_pages']} pages "
                f"in {format_time(extract_time)}"
            )
            
            # Check if we actually got any text
            if pdf_data["total_chars"] == 0:
                st.error(
                    "❌ No text could be extracted from this PDF. "
                    "It might be a scanned document (image-based)."
                )
                st.stop()
            
            # ── Step 3: Split text into chunks ──
            status_text.text("✂️ Splitting text into chunks...")
            progress_bar.progress(40, text="Chunking text...")
            
            start = time.time()
            chunk_data = split_text_into_chunks(pdf_data["full_text"])
            chunk_time = time.time() - start
            
            add_log(
                f"Created {chunk_data['num_chunks']} chunks "
                f"(avg {chunk_data['avg_chunk_size']} chars) "
                f"in {format_time(chunk_time)}"
            )
            
            # ── Step 4: Load embedding model ──
            status_text.text("🧠 Loading embedding model...")
            progress_bar.progress(55, text="Loading model...")
            
            start = time.time()
            embeddings_model = load_embedding_model()
            model_time = time.time() - start
            
            add_log(f"Embedding model loaded in {format_time(model_time)}")
            
            # ── Step 5: Try to load existing index or create new one ──
            # Check if we have a saved index for this exact file
            hash_file = os.path.join(index_path, "file_hash.txt")
            saved_hash = None
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    saved_hash = f.read().strip()
            
            if saved_hash == file_hash:
                # The file hasn't changed, try loading the saved index
                status_text.text("📂 Loading saved index...")
                progress_bar.progress(75, text="Loading saved index...")
                
                start = time.time()
                vector_store = load_vector_store(index_path, embeddings_model)
                index_time = time.time() - start
                
                if vector_store is not None:
                    add_log(
                        f"Loaded saved FAISS index in {format_time(index_time)}"
                    )
                else:
                    add_log(
                        "Saved index was missing or corrupted. Rebuilding index."
                    )
                    status_text.text("🔢 Rebuilding missing index...")
                    progress_bar.progress(70, text="Building vector index...")
                    
                    start = time.time()
                    vector_store = create_vector_store(
                        chunk_data["chunks"], embeddings_model
                    )
                    embed_time = time.time() - start
                    
                    add_log(
                        f"Created FAISS index with {chunk_data['num_chunks']} "
                        f"vectors in {format_time(embed_time)}"
                    )
                    
                    status_text.text("💾 Saving index to disk...")
                    progress_bar.progress(90, text="Saving index...")
                    
                    save_vector_store(vector_store, index_path)
                    
                    os.makedirs(index_path, exist_ok=True)
                    with open(hash_file, "w") as f:
                        f.write(file_hash)
                    
                    add_log("Saved FAISS index to disk")
            else:
                # New file or file changed, create fresh index
                status_text.text("🔢 Generating embeddings & building index...")
                progress_bar.progress(70, text="Building vector index...")
                
                start = time.time()
                vector_store = create_vector_store(
                    chunk_data["chunks"], embeddings_model
                )
                embed_time = time.time() - start
                
                add_log(
                    f"Created FAISS index with {chunk_data['num_chunks']} "
                    f"vectors in {format_time(embed_time)}"
                )
                
                # Save the index for next time
                status_text.text("💾 Saving index to disk...")
                progress_bar.progress(90, text="Saving index...")
                
                save_vector_store(vector_store, index_path)
                
                os.makedirs(index_path, exist_ok=True)
                with open(hash_file, "w") as f:
                    f.write(file_hash)
                
                add_log("Saved FAISS index to disk")
            
            # ── Step 6: Store everything in session state ──
            st.session_state.vector_store = vector_store
            st.session_state.doc_processed = True
            st.session_state.doc_info = {
                "file_name": uploaded_file.name,
                "num_pages": pdf_data["num_pages"],
                "total_chars": pdf_data["total_chars"],
                "num_chunks": chunk_data["num_chunks"],
                "avg_chunk_size": chunk_data["avg_chunk_size"],
            }
            
            progress_bar.progress(100, text="Done!")
            status_text.text("")
            add_log("✅ Document processing complete!")
            
            st.success("✅ Document processed successfully! You can now ask questions.")
            time.sleep(1.5)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            add_log(f"ERROR: {str(e)}")


# ──────────────────────────────────────────────
# Main page - Title and Description
# ──────────────────────────────────────────────
st.title("📚 AutoInsight — AI-Powered Document QA")
st.markdown(
    "Upload a PDF document and ask questions about it. "
    "The system retrieves relevant sections and uses **Google Gemini** "
    "to generate accurate answers."
)

st.markdown("---")


# ──────────────────────────────────────────────
# Question Input and Answer Display
# ──────────────────────────────────────────────
if st.session_state.doc_processed:
    # Question input area
    user_question = st.text_input(
        "🔍 Ask a question about your document:",
        placeholder="e.g., What is the main topic of this document?",
        key="question_input",
    )
    
    # Handle question submission
    if user_question:
        with st.spinner("🔍 Searching document and generating answer..."):
            try:
                # Time the retrieval and generation separately
                start_retrieval = time.time()
                
                # Run the full RAG pipeline
                result = ask_question(
                    st.session_state.vector_store,
                    user_question,
                    num_chunks=4,
                )
                
                total_time = time.time() - start_retrieval
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": result["answer"],
                    "context_chunks": result["context_chunks"],
                    "scores": result["similarity_scores"],
                    "time": total_time,
                })
                
                add_log(
                    f"Answered question in {format_time(total_time)}"
                )
                
            except ValueError as e:
                # This usually means the API key is missing
                st.error(f"🔑 API Key Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    
    # Display chat history (most recent first)
    if st.session_state.chat_history:
        st.markdown("---")
        
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            chat_num = len(st.session_state.chat_history) - i
            
            # Question
            st.markdown(f"**🙋 Question {chat_num}:** {entry['question']}")
            
            # Answer
            st.markdown("**🤖 Answer:**")
            st.markdown(entry["answer"])
            
            # Timing info
            st.caption(f"⏱️ Total time: {format_time(entry['time'])}")
            
            # Retrieved context (expandable)
            with st.expander(f"📄 Retrieved Context (Top {len(entry['context_chunks'])} chunks)"):
                for j, (chunk, score) in enumerate(
                    zip(entry["context_chunks"], entry["scores"])
                ):
                    st.markdown(f"**Chunk {j + 1}** (similarity score: {score})")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    st.markdown("---")
            
            st.markdown("---")
else:
    # Show instructions if no document is processed yet
    st.info(
        "👈 **Upload a PDF** in the sidebar and click "
        "**Process Document** to get started!"
    )
    
    # Show a brief explanation of how RAG works
    st.markdown("### How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Upload")
        st.markdown(
            "Upload your PDF document. The text is extracted "
            "and split into small chunks."
        )
    
    with col2:
        st.markdown("#### 2️⃣ Process")
        st.markdown(
            "Each chunk is converted into a numerical vector "
            "(embedding) and stored in a FAISS index."
        )
    
    with col3:
        st.markdown("#### 3️⃣ Ask")
        st.markdown(
            "Ask a question! The system finds the most relevant "
            "chunks and uses Gemini to generate an answer."
        )


# ──────────────────────────────────────────────
# Processing Log (at the bottom)
# ──────────────────────────────────────────────
if st.session_state.processing_log:
    with st.expander("📋 Processing Log"):
        for log_entry in st.session_state.processing_log:
            st.text(log_entry)
