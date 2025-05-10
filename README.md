## Key Design Choices
...
*   **HuggingFace Embeddings (`all-MiniLM-L6-v2`):** A lightweight, fast, and effective open-source embedding model from Sentence Transformers, running locally. This significantly reduces the project's footprint compared to larger models.
...

## How to Run
...
4.  **Ingest Data:**
    *   Place `.txt` documents in `documents/`.
    *   Run ingestion (this will download the `all-MiniLM-L6-v2` embedding model, approx. 80MB, on first run):
        ```bash
        python ingest.py
        ```
    *   **Important:** If you previously ran `ingest.py` with a different embedding model, the script will attempt to delete the old `vector_store/` directory to avoid compatibility issues.
...