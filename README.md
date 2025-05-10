# RAG-Powered Multi-Agent Q&A Assistant (Groq & MiniLM)

This Streamlit application provides a multi-agent question-answering system. It can answer questions based on documents you provide (via RAG), perform calculations, and define words. It leverages Groq's fast LLM (Llama3-8b) and a local sentence-transformer model (all-MiniLM-L6-v2) for embeddings.

## Architecture

The application is a single Python file (`app.py`) designed for easy deployment on Streamlit Cloud.

1.  **Streamlit UI**: Provides the web interface for user interaction (chat input, displaying responses, agent thoughts, and retrieved context).
2.  **Langchain Agent (ReAct)**: An agent that decides which tool to use based on the user's query. It uses the ReAct (Reasoning and Acting) framework.
3.  **Tools**:
    *   **KnowledgeBaseSearch (RAG Tool)**:
        *   Retrieves relevant text chunks from a local vector store (ChromaDB) built from documents in the `documents/` directory.
        *   Uses `sentence-transformers/all-MiniLM-L6-v2` for creating text embeddings.
        *   Passes the retrieved context and the query to the LLM to generate an answer.
    *   **Calculator**: Uses the LLM to evaluate mathematical expressions.
    *   **Dictionary**: Fetches word definitions from an external API (`dictionaryapi.dev`).
4.  **LLM (Groq - Llama3-8b-8192)**: The core language model used for generating responses, powering the agent's reasoning, and the calculator tool.
5.  **Vector Store (ChromaDB)**:
    *   Stores embeddings of document chunks.
    *   Created on-the-fly within the `vector_store/` directory if it doesn't exist or seems invalid.
    *   The `documents/` directory must be populated with `.txt` files for ingestion.
6.  **Embedding Model (HuggingFace - all-MiniLM-L6-v2)**: A lightweight and efficient model used to convert text documents and queries into numerical vectors for similarity search.

## Key Design Choices

*   **All-in-One `app.py`**: The entire application logic, including data ingestion, is contained within a single `app.py` file. This simplifies deployment on platforms like Streamlit Cloud, which typically run a single entry-point script and require all dependencies to be managed within the app's environment.
*   **On-the-Fly Data Ingestion & Local Vector Store**:
    *   The application checks for a `vector_store/` directory on startup.
    *   If the vector store is missing, empty, or appears invalid, the application automatically ingests `.txt` documents from a local `documents/` folder.
    *   This design avoids reliance on external databases and allows the application to be self-contained, making it suitable for environments where persistent, managed vector databases are not readily available. ChromaDB is used for its simplicity in local file-based persistence.
*   **Lightweight Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` is chosen for its balance of performance and small size, making it suitable for CPU-based inference and quicker startup/ingestion times within resource-constrained environments like Streamlit Cloud's free tier.
*   **Fast LLM via Groq**: Llama3-8b-8192 hosted on Groq provides very fast inference speeds, crucial for a responsive chat experience.
*   **Modular Tools**: The agent uses distinct tools for different tasks (RAG, calculation, dictionary), allowing for specialized handling and better accuracy for each task.
*   **Graceful Degradation**: If the vector store cannot be initialized (e.g., no documents or ingestion error), the RAG tool (`KnowledgeBaseSearch`) is disabled, but the Calculator and Dictionary tools remain functional.
*   **Streamlit Caching**: `@st.cache_resource` is used to cache the loaded vector store and embedding model, preventing re-initialization on every user interaction and improving performance.

## How to Run

1.  **Prerequisites**:
    *   Python 3.8+
    *   `pip` for package management

2.  **Create Project Directory and `documents` folder**:
    ```bash
    mkdir my_rag_app
    cd my_rag_app
    mkdir documents
    ```
    Place your `.txt` source documents inside the `documents/` folder.

3.  **Create `requirements.txt`**:
    Create a file named `requirements.txt` in your project directory with the following content:
    ```
    streamlit
    python-dotenv
    langchain
    langchain-groq
    langchain-community
    sentence-transformers
    chromadb
    requests
    pydantic>=1,<2
    faiss-cpu # Optional but often a dependency for vector stores or if chromadb needs it
    ```
    *Note: `pydantic` version is pinned to `<2` as some Langchain components might still have compatibility issues with Pydantic v2. Adjust if needed.*

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Environment Variables**:
    You need a Groq API key.
    *   **For local development**: Create a `.env` file in your project directory:
        ```
        GROQ_API_KEY_LOCAL="your_actual_groq_api_key"
        ```
    *   **For Streamlit Cloud deployment**: Set the `GROQ_API_KEY` as a secret in your Streamlit Cloud app settings.

6.  **Save the Code**:
    Save the provided Python code as `app.py` in your project directory.

7.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

The application will start, and if no `vector_store` exists, it will attempt to ingest documents from the `documents` folder. You can then interact with the Q&A assistant through your web browser.