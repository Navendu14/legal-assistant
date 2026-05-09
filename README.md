# ⚖️ Legal AI Assistant

A multi-user, web-based conversational AI legal assistant built with Streamlit and LangChain. This application uses a Retrieval-Augmented Generation (RAG) architecture to answer legal questions accurately by referencing a localized vector database of ingested legal documents.

## 🚀 Key Features

- **ReAct Agent Architecture**: Powered by LangChain and Google's Gemini LLM to reason through legal queries and fetch documents before responding, preventing AI hallucinations.
- **RAG Pipeline**: Uses ChromaDB and HuggingFace sentence-transformers to index, search, and retrieve context from local PDF documents.
- **User Authentication**: Custom built-in login and signup system with passwords securely hashed using SHA-256 (stored via SQLite).
- **Persistent Chat Memory**: Retains user-specific conversational context and history across sessions, stored in SQLite databases.
- **Interactive UI**: Clean, responsive frontend built entirely with Streamlit.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (`gemini-2.5-flash-lite`)
- **Orchestration**: LangChain (ReAct Agents, Tools, Memory)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`sentence-transformers/all-mpnet-base-v2`)
- **Relational Database**: SQLite3 (Users & Chat History)

## 📋 Prerequisites

- Python 3.9+
- A Google Gemini API Key

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd legal-assistant
   ```

2. **Install dependencies:**
   Ensure you have the required libraries installed. You can install them via pip:
   ```bash
   pip install streamlit langchain langchain-google-genai langchain-chroma langchain-huggingface pyyaml python-dotenv pypdf
   ```

3. **Configuration:**
   Create a `config.yaml` file in the root directory with the following structure:
   ```yaml
   GOOGLE_API_KEY: "your_google_api_key_here"
   data_dir: "./data"                  # Directory containing your legal PDFs
   chunk_size: 1000
   chunk_overlap: 200
   batch_size: 10
   embedding_model: "sentence-transformers/all-mpnet-base-v2"
   persist_directory: "./chroma_db"
   collection_name: "Legal_document"
   chunk_batch_size: 50
   ```

## 📚 Usage

### 1. Data Ingestion (Admin)
Before using the chatbot, you must ingest your legal PDF documents into the vector database. Place your PDFs in the directory specified by `data_dir` in your config, then run:

```bash
python ingestion.py
```
This will parse the PDFs, generate vector embeddings, and save them locally to `./chroma_db`.

### 2. Run the Web Application
Start the Streamlit application:

```bash
streamlit run app3.py
```
This will open the web interface in your browser where you can create an account, log in, and start asking legal questions.

### 3. Run the CLI Application (Optional)
If you prefer testing the agent in your terminal without the UI, run:
```bash
python rag.py
```

## 🗂️ Project Structure

- `app3.py`: The main Streamlit web application.
- `ingestion.py`: Script to process PDFs and populate the ChromaDB vector store.
- `rag.py`: A terminal-based version of the chatbot for CLI interaction.
- `auth_db.py`: Handles SQLite user database initialization, password hashing, and authentication.
- `db.py`: Handles SQLite chat history database initialization and message storage.
