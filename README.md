# PDF Semantic Search (RAG) Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their contents using natural language.

The application processes uploaded documents, creates vector embeddings, retrieves relevant sections based on the user's query, and generates answers using the retrieved context.

## Features

* 📄 Upload PDF documents
* 📚 Support for multiple documents
* 🔍 Semantic search over document content
* 💬 Ask questions using natural language
* 🧩 Retrieval-Augmented Generation (RAG) pipeline
* 🗂️ Vector storage and similarity search using ChromaDB
* 🖥️ Simple interactive interface built with Streamlit

## Tech Stack

* **Python** — Application and RAG pipeline
* **LangChain** — Document processing and retrieval pipeline
* **ChromaDB** — Vector database for storing and searching embeddings
* **HuggingFace** — Text embedding generation
* **Streamlit** — Web interface

## How It Works

The chatbot follows a basic RAG pipeline:

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Chunking
      ↓
HuggingFace Embeddings
      ↓
ChromaDB Vector Store
      ↓
User Query
      ↓
Semantic Similarity Search
      ↓
Relevant Document Chunks
      ↓
Answer Generation
```

### 1. Document Upload

Users upload one or more PDF documents through the Streamlit interface.

### 2. Text Processing

The documents are loaded and split into smaller chunks so that relevant sections can be retrieved efficiently.

### 3. Embedding Generation

Each document chunk is converted into a numerical vector using a HuggingFace embedding model.

### 4. Vector Storage

The generated embeddings are stored in ChromaDB, which allows the application to perform semantic similarity searches.

### 5. Querying

When a user asks a question, the query is converted into an embedding and compared against the stored document embeddings.

The most relevant chunks are retrieved and used as context for answering the question.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## Example Use Cases

The chatbot can be used to query documents such as:

* Study materials
* Research papers
* Technical documentation
* Books and notes
* Company documents

For example:

> Upload a machine learning textbook and ask:
> **"What is the difference between supervised and unsupervised learning?"**

The system retrieves relevant sections from the uploaded document and uses them to answer the question.

## Limitations

This is a learning project demonstrating the fundamentals of Retrieval-Augmented Generation and semantic search.

The quality of answers depends on factors such as document structure, text extraction, chunking strategy, embedding quality, and retrieval results.

## Future Improvements

* Add conversation history
* Add support for more document formats
* Experiment with different embedding models
* Add retrieval evaluation and benchmarking
* Improve chunking strategies
* Add metadata filtering
* Add document management and persistent storage

## License

This project is available for educational and personal use.
