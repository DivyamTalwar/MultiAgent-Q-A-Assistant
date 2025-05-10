import streamlit as st
import os
import requests
import re
import shutil
import traceback
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler

DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_GROQ = "llama3-8b-8192"

_EMBEDDINGS_MODEL = None

def get_embeddings_model():
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        print(f"Initializing HuggingFace embeddings model (once): {EMBEDDING_MODEL_NAME}...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model initialized globally for the app.")
    return _EMBEDDINGS_MODEL

def ingest_data_st():
    st.info(f"Starting data ingestion process...")
    st.info(f"Looking for documents in ./{DOCUMENTS_DIR}/")

    if not os.path.exists(DOCUMENTS_DIR):
        st.error(f"The '{DOCUMENTS_DIR}' directory was not found. Please create it and add your .txt files.")
        return False
    
    doc_paths = [os.path.join(DOCUMENTS_DIR, fname) for fname in os.listdir(DOCUMENTS_DIR) if fname.endswith(".txt")]

    if not doc_paths:
        st.warning(f"No .txt documents found in '{DOCUMENTS_DIR}'. Cannot create vector store.")
        if os.path.exists(VECTOR_STORE_DIR):
            try:
                shutil.rmtree(VECTOR_STORE_DIR)
            except Exception as e:
                st.warning(f"Could not remove existing vector store directory: {e}")
        return False

    documents = []
    for doc_path in doc_paths:
        try:
            loader = TextLoader(doc_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading document {doc_path}: {e}")
            continue

    if not documents:
        st.error("No documents were successfully loaded. Ingestion cannot proceed.")
        return False
    st.info(f"Loaded {len(documents)} document(s).")

    st.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    st.info(f"Created {len(chunks)} chunks.")

    st.info(f"Ensuring HuggingFace embeddings model ({EMBEDDING_MODEL_NAME}) is ready...")
    embeddings_model_instance = get_embeddings_model()
    st.info("Embedding model is ready.")

    st.info(f"Creating vector store in ./{VECTOR_STORE_DIR}/ using ChromaDB...")
    if os.path.exists(VECTOR_STORE_DIR):
        st.info(f"Removing old vector store at ./{VECTOR_STORE_DIR}/ to ensure freshness.")
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
        except Exception as e:
            st.error(f"Error removing old vector store: {e}. Proceeding with caution.")
    
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
    
    try:
        vector_store_instance = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model_instance,
            persist_directory=VECTOR_STORE_DIR
        )
        vector_store_instance.persist()
        st.success("Vector store created and persisted successfully!")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

@st.cache_resource
def load_vector_store_and_ingest_if_needed():
    embeddings_func = get_embeddings_model()

    is_valid_store = False
    if os.path.exists(VECTOR_STORE_DIR) and os.path.isdir(VECTOR_STORE_DIR):
        if any(f.endswith('.sqlite3') for f in os.listdir(VECTOR_STORE_DIR)):
            is_valid_store = True

    if not is_valid_store:
        st.warning(f"Vector store not found or appears invalid at ./{VECTOR_STORE_DIR}/. Attempting to ingest data...")
        with st.spinner("Ingesting data... This may take a few minutes. Please wait."):
            ingestion_success = ingest_data_st()
        if not ingestion_success:
            st.error("Data ingestion failed. RAG functionality will be unavailable.")
            return None
    else:
        st.info(f"Found existing vector store at ./{VECTOR_STORE_DIR}/. Loading...")

    try:
        vs = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_func)
        try:
            vs.similarity_search("test query", k=1)
            st.success("Vector store loaded successfully.")
            return vs
        except Exception as e:
            st.warning(f"Loaded vector store at ./{VECTOR_STORE_DIR}/ seems problematic or empty ({str(e)[:100]}...). Attempting to re-ingest data.")
            with st.spinner("Re-ingesting data due to potential issue... Please wait."):
                ingestion_success = ingest_data_st()
            if not ingestion_success:
                st.error("Data re-ingestion failed. RAG functionality will be unavailable.")
                return None
            vs = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_func)
            st.success("Vector store re-ingested and loaded successfully.")
            return vs
    except Exception as e:
        st.error(f"Failed to load or re-initialize vector store from ./{VECTOR_STORE_DIR}/: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

st.set_page_config(page_title="RAG Multi-Agent Q&A (Groq & MiniLM)", layout="wide")

load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in Streamlit secrets (GROQ_API_KEY) or as GROQ_API_KEY_LOCAL in your .env for local testing.")
    st.stop()

llm = ChatGroq(temperature=0, model_name=LLM_MODEL_GROQ, groq_api_key=GROQ_API_KEY)
vector_store = load_vector_store_and_ingest_if_needed()

rag_tool = None
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    def run_rag_tool(query: str):
        try:
            result = rag_chain.invoke({"query": query})
            st.session_state.retrieved_contexts = [doc.page_content for doc in result['source_documents']]
            return result['result']
        except Exception as e:
            return f"Error during RAG search: {e}"

    rag_tool = Tool(
        name="KnowledgeBaseSearch",
        func=run_rag_tool,
        description="Use this tool to answer questions about company policies, product specifications, FAQs, and general knowledge found in the provided documents. Input should be a complete question."
    )
else:
    st.warning("KnowledgeBaseSearch tool is unavailable because the vector store could not be initialized.")

class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate. E.g., '2 + 2', 'sqrt(25) * 3'")

def calculate_expression(expression: str):
    try:
        calculator_prompt = f"Evaluate the following mathematical expression and return only the final numerical result (no explanation, just the number itself, ensure it is a valid number): {expression}"
        response = llm.invoke(calculator_prompt)
        answer = response.content.strip()
        
        match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', answer)
        if match:
            return match.group(0).replace(',', '')
        
        parts = answer.split()
        for part in reversed(parts):
            try:
                num_part = part.replace(',', '').replace('$', '').replace('€', '').strip()
                if re.fullmatch(r'-?\d+(?:\.\d+)?', num_part):
                    return str(float(num_part))
            except ValueError:
                continue
        return f"Could not extract a clear numerical answer. Raw model output: {answer}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

calculator_tool = Tool.from_function(
    func=calculate_expression,
    name="Calculator",
    description="Use this tool when you need to answer questions involving mathematical calculations or to evaluate a math expression. Input should be the mathematical expression itself.",
    args_schema=CalculatorInput
)

class DictionaryInput(BaseModel):
    word: str = Field(description="The single word to define.")

def define_word(word: str):
    try:
        cleaned_word = word.lower().strip()
        if not cleaned_word:
            return "Please provide a word to define."
        if ' ' in cleaned_word:
            return f"The dictionary tool works best with single words. You provided: '{word}'."
        
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{cleaned_word}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data and isinstance(data, list) and data[0].get('meanings'):
            definitions = []
            for meaning in data[0]['meanings'][:2]:
                for def_obj in meaning.get('definitions', [])[:2]:
                    definitions.append(f"- ({meaning.get('partOfSpeech', 'N/A')}) {def_obj.get('definition', 'No definition text.')}")
            if definitions:
                return "\n".join(definitions)
            return f"No definitions found for '{word}' in the expected response structure."
        elif isinstance(data, dict) and data.get("title") == "No Definitions Found":
             return f"Could not find a definition for '{word}' in the dictionary (API: No Definitions Found)."
        return f"No definition found for '{word}' or API response format unexpected. Response: {data}"
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return f"Could not find a definition for '{word}' in the dictionary (404 Not Found)."
        return f"Dictionary API HTTP error: {http_err} for word '{word}'. URL: {http_err.request.url if http_err.request else 'N/A'}"
    except requests.exceptions.RequestException as req_err:
        return f"Dictionary API request error: {req_err} for word '{word}'"
    except Exception as e:
        return f"An unexpected error occurred while fetching definition for '{word}': {str(e)}. Type: {type(e)}"

dictionary_tool = Tool.from_function(
    func=define_word,
    name="Dictionary",
    description="Use this tool when you need to find the definition of a specific word. Input should be the single word you want defined.",
    args_schema=DictionaryInput
)

tools = [calculator_tool, dictionary_tool]
if rag_tool:
    tools.insert(0, rag_tool)

class AgentCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs) -> any:
        thought = action.log.strip()
        tool_info = ""
        if action.tool and action.tool_input:
            tool_info = f"➡️ Agent chose tool: **{action.tool}** with input: **{action.tool_input}**"
        log_message = f"🤔 Thought: {thought}\n{tool_info}"
        if "agent_logs" in st.session_state:
            st.session_state.agent_logs.append(log_message)

    def on_tool_end(self, output, name, **kwargs) -> any:
        pass

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="I'm having trouble understanding the next step, can you rephrase or try a different approach? If a tool is not working, I will state that.",
    max_iterations=7,
)

st.title(" RAG-Powered Multi-Agent Q&A Assistant (Groq & MiniLM)")
st.markdown("""
Ask questions about our documents (e.g., "Business hours?"), request calculations (e.g., "Calc 100/5 + 3"), or ask for word definitions (e.g., "Define ephemeral").
The agent will use Groq's fast LLMs and a lightweight local embedding model to respond.
**Note:** Data ingestion from the `./documents` folder happens on first load or if the vector store is missing/invalid.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "retrieved_contexts" not in st.session_state:
    st.session_state.retrieved_contexts = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.agent_logs = []
    st.session_state.retrieved_contexts = []
    agent_callback_handler = AgentCallbackHandler()

    with st.chat_message("assistant"):
        with st.spinner("Groq-cessing with MiniLM... Please wait..."):
            try:
                response = agent_executor.invoke(
                    {"input": prompt, "chat_history": []},
                    {"callbacks": [agent_callback_handler]}
                )
                answer = response['output']

                if st.session_state.agent_logs:
                    st.subheader("Agent Decision Path & Thoughts:")
                    for log_entry in st.session_state.agent_logs:
                        st.markdown(log_entry)
                else:
                    st.markdown("*Agent may have answered directly or detailed logs are in console (verbose=True).*")
                
                if st.session_state.retrieved_contexts:
                    st.subheader("Retrieved Context Snippets (if RAG used):")
                    for i, context in enumerate(st.session_state.retrieved_contexts):
                        with st.expander(f"Context Snippet {i+1}"):
                            st.text(context)
                
                st.subheader("Final Answer:")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                detailed_traceback = traceback.format_exc()
                st.error(error_message)
                st.error(f"Traceback:\n```\n{detailed_traceback}\n```")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})