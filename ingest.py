import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_data():
    print(f"Loading documents from {DOCUMENTS_DIR}...")
    doc_paths = [os.path.join(DOCUMENTS_DIR, fname) for fname in os.listdir(DOCUMENTS_DIR) if fname.endswith(".txt")]
    documents = []
    for doc_path in doc_paths:
        loader = TextLoader(doc_path, encoding='utf-8')
        documents.extend(loader.load())

    if not documents:
        print("No documents found. Exiting.")
        return
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Initializing HuggingFace embeddings model: {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model initialized.")

    print(f"Creating vector store in {VECTOR_STORE_DIR} using ChromaDB...")
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
    
    if os.path.exists(VECTOR_STORE_DIR):
        import shutil
        print(f"Removing old vector store at {VECTOR_STORE_DIR} to ensure compatibility with new embedding model.")
        shutil.rmtree(VECTOR_STORE_DIR)
        os.makedirs(VECTOR_STORE_DIR)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=VECTOR_STORE_DIR
    )
    vector_store.persist()
    print("Vector store created and persisted successfully.")

if __name__ == "__main__":
    ingest_data()