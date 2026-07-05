# Document Question Answering System (RAG)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that answers questions based on uploaded PDF documents. The system retrieves the most relevant sections from your document and uses Google Gemini to generate accurate answers.

---

## 🏗️ Architecture

```
Upload PDF
      │
      ▼
Extract Text (PyPDF2)
      │
      ▼
Split into Chunks (LangChain RecursiveCharacterTextSplitter)
      │
      ▼
Generate Embeddings (HuggingFace all-MiniLM-L6-v2)
      │
      ▼
Store in FAISS Vector Database
      │
─────────────────────────────────────
      │
User asks a Question
      │
      ▼
Convert Question to Embedding
      │
      ▼
Retrieve Top 4 Similar Chunks (FAISS)
      │
      ▼
Build Prompt with Context + Question
      │
      ▼
Send to Google Gemini API
      │
      ▼
Display Answer to User
```

---

## 📂 Project Structure

```
RAG_Document_QA/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env                      # API key configuration
│
├── data/                     # Uploaded PDFs and FAISS indexes
│
├── modules/
│   ├── pdf_loader.py         # PDF text extraction
│   ├── text_splitter.py      # Text chunking logic
│   ├── embedding_store.py    # Embeddings and FAISS vector store
│   └── rag_pipeline.py       # RAG pipeline (retrieval + generation)
│
└── utils/
    └── helper.py             # Utility functions
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- A Google Gemini API key (free at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Steps

1. **Clone or download this project**

```bash
cd RAG_Document_QA
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up your API key**

Open the `.env` file and replace the placeholder with your actual Gemini API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

5. **Run the application**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ How to Use

1. **Upload a PDF** — Use the sidebar to upload your PDF document.
2. **Click Process Document** — The system extracts text, creates chunks, generates embeddings, and builds the search index.
3. **Ask Questions** — Type your question in the input box and get AI-generated answers based on your document.
4. **View Context** — Expand the "Retrieved Context" section to see which parts of the document were used to generate the answer.

---

## 📸 Screenshots

*Screenshots will be added here after running the application.*

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Programming language |
| **Streamlit** | Web interface / frontend |
| **LangChain** | Text splitting and retrieval pipeline |
| **FAISS** | Local vector database for similarity search |
| **HuggingFace Transformers** | Local embedding generation (all-MiniLM-L6-v2) |
| **Google Gemini** | LLM for answer generation |
| **PyPDF2** | PDF text extraction |

---

## ✨ Features

- 📄 PDF document upload and text extraction
- ✂️ Smart text chunking with overlap
- 🧠 Local embedding generation (no cloud APIs needed for embeddings)
- 🔍 Fast similarity search with FAISS
- 💬 Natural language question answering with Google Gemini
- 💾 Cached FAISS indexes (no reprocessing needed for the same file)
- ⏱️ Timing information for retrieval and generation
- 📋 Processing logs
- 🗑️ Clear chat functionality
- 📊 Document statistics (pages, characters, chunks)

---

## 🔮 Future Improvements

- Support for multiple file formats (DOCX, TXT, etc.)
- Support for uploading multiple documents at once
- Chat memory to handle follow-up questions
- Highlighting the exact source passages in the PDF
- Better handling of scanned/image-based PDFs using OCR
- Option to choose different LLM models
- Export chat history as a report
- Add authentication for multi-user support

---

## ⚠️ Important Notes

- This project uses **FAISS** as the vector database (not Pinecone or any cloud vector DB).
- Embeddings are generated **locally** using HuggingFace models.
- Only **Google Gemini** is used for text generation.
- The application works entirely on **local document embeddings** — your data stays on your machine.
- You need a valid **Google Gemini API key** to use the question-answering feature.

---

## 📝 License

This project is for educational purposes. Feel free to use and modify it for learning.
