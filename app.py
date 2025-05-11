import streamlit as st
import os
import requests
import re
import shutil
import traceback
import sys # Required for the SQLite patch
from dotenv import load_dotenv

# --- BEGIN SQLITE PATCH FOR CHROMADB ---
# This patch is to address SQLite version issues on Streamlit Cloud for ChromaDB.
# It attempts to use 'pysqlite3-binary' if available.
# Ensure 'pysqlite3-binary' is added to your requirements.txt file.
_sqlite_patch_applied_message = ""
_patch_successful = False
try:
    # Attempt to import pysqlite3 and then swap it into sys.modules
    __import__('pysqlite3')
    # If the import was successful, pysqlite3 is now in sys.modules
    # We now want to make 'sqlite3' in sys.modules point to the pysqlite3 module
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    _sqlite_patch_applied_message = "✅ Using bundled SQLite version for ChromaDB compatibility."
    _patch_successful = True
    # Optional: Verify in logs (will appear in Streamlit Cloud logs)
    import sqlite3
    print(f"SQLite patch successful. Using SQLite version: {sqlite3.sqlite_version}")

except ImportError:
    _sqlite_patch_applied_message = "⚠️ 'pysqlite3-binary' not found. Using system SQLite. If ChromaDB errors persist, add 'pysqlite3-binary' to requirements.txt."
    print("SQLite patch: pysqlite3-binary not found. ChromaDB will use system's sqlite3.")
except Exception as e:
    _sqlite_patch_applied_message = f"❌ Error applying SQLite patch: {e}"
    print(f"SQLite patch: An error occurred during patch attempt: {e}")
# --- END SQLITE PATCH FOR CHROMADB ---

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # Chroma import happens AFTER the patch
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
    if _patch_successful and "vector_store" not in st.session_state.get("patch_success_message_shown", []):
        st.toast(_sqlite_patch_applied_message, icon="🚀") # Show success toast once during ingestion
        if "patch_success_message_shown" not in st.session_state:
            st.session_state.patch_success_message_shown = []
        st.session_state.patch_success_message_shown.append("vector_store")


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
        st.error(f"Traceback: {traceback.format_exc()}") # This will show the sqlite error if patch failed
        if not _patch_successful and "pysqlite3-binary" not in _sqlite_patch_applied_message : # Check if it's the import error
             st.warning("The SQLite patch might be missing. Ensure 'pysqlite3-binary' is in your requirements.txt and the app is restarted.")
        return False

@st.cache_resource
def load_vector_store_and_ingest_if_needed():
    # Display patch status message from the UI, once.
    # This is a bit early for a toast, but good for a caption or sidebar.
    # Moved toast to ingest_data_st for better timing.
    # If patch was not successful and it wasn't an import error, show the message
    if not _patch_successful and "pysqlite3-binary" in _sqlite_patch_applied_message : # i.e. import error
        if "patch_fail_message_shown" not in st.session_state:
            st.toast(_sqlite_patch_applied_message, icon="⚠️")
            st.session_state.patch_fail_message_shown = True # Show only once per session
    elif not _patch_successful: # Other patch error
        if "patch_error_message_shown" not in st.session_state:
            st.toast(_sqlite_patch_applied_message, icon="❌")
            st.session_state.patch_error_message_shown = True


    embeddings_func = get_embeddings_model()

    is_valid_store = False
    if os.path.exists(VECTOR_STORE_DIR) and os.path.isdir(VECTOR_STORE_DIR):
        # A more robust check could be to see if essential Chroma files exist, e.g., a .sqlite3 file
        if any(f.endswith('.sqlite3') for f in os.listdir(VECTOR_STORE_DIR)): # Chroma typically creates a .sqlite3 file
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
        # Test the connection/validity of the loaded vector store
        try:
            vs.similarity_search("test query", k=1) # A simple query to check if it works
            st.success("Vector store loaded successfully.")
            return vs
        except Exception as e:
            st.warning(f"Loaded vector store at ./{VECTOR_STORE_DIR}/ seems problematic or empty ({str(e)[:100]}...). Attempting to re-ingest data.")
            # This could be where the old sqlite version causes issues if the patch didn't work and an old DB exists
            with st.spinner("Re-ingesting data due to potential issue... Please wait."):
                ingestion_success = ingest_data_st()
            if not ingestion_success:
                st.error("Data re-ingestion failed. RAG functionality will be unavailable.")
                return None
            # Reload after re-ingestion
            vs = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_func)
            st.success("Vector store re-ingested and loaded successfully.")
            return vs
    except Exception as e:
        st.error(f"Failed to load or re-initialize vector store from ./{VECTOR_STORE_DIR}/: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        if not _patch_successful:
             st.warning("This failure might be related to SQLite version. Ensure 'pysqlite3-binary' is in requirements.txt and the patch was applied.")
        return None

st.set_page_config(page_title="RAG Multi-Agent Q&A (Groq & MiniLM)", layout="wide")

# Display the SQLite patch status message (can be a caption or toast)
# Using st.caption for a less intrusive startup message
# st.caption(_sqlite_patch_applied_message) # This is also an option for placement

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except (KeyError, AttributeError):
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in Streamlit secrets (GROQ_API_KEY) or as GROQ_API_KEY_LOCAL in your .env for local testing.")
    st.stop()

llm = ChatGroq(temperature=0, model_name=LLM_MODEL_GROQ, groq_api_key=GROQ_API_KEY)
vector_store = load_vector_store_and_ingest_if_needed() # This now runs after the patch attempt

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
    # If vector_store is None, and patch was not successful, reiterate the patch message.
    if not _patch_successful and "vector_store_unavailable_patch_warn" not in st.session_state:
        st.warning(f"Note on compatibility: {_sqlite_patch_applied_message}")
        st.session_state.vector_store_unavailable_patch_warn = True


class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate. E.g., '2 + 2', 'sqrt(25) * 3'")

def calculate_expression(expression: str):
    try:
        # Using a more robust way to ask LLM for calculation, instructing it to be a calculator
        calculator_prompt = f"Please evaluate the following mathematical expression and return only the final numerical result. Ensure the result is a single, clean number (e.g., '23.5', '-1000', '0.05'). Do not include any explanations, units, or currency symbols unless they are part of a standard mathematical notation requested in the expression. Expression: {expression}"
        response = llm.invoke(calculator_prompt)
        answer = response.content.strip()
        
        # Attempt to extract a number using a more general regex, including scientific notation
        # This regex looks for numbers like: 123, 123.45, -123, -123.45, 1.23e+5, -1.23E-5
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', answer)
        
        if match:
            extracted_num_str = match.group(0)
            try:
                # Validate if it can be converted to float
                float(extracted_num_str) 
                return extracted_num_str
            except ValueError:
                # If conversion fails, it might be something like "1,234.56" which re.search might pick partially
                # or the LLM included text. Fallback to trying to clean common non-numeric parts.
                pass # Fall through to further processing

        # If regex fails or gives a non-numeric string, try cleaning common issues from LLM output
        # Remove common textual artifacts if any, like "The result is", "Answer:", etc.
        # This part is tricky as LLMs can be verbose. The prompt above tries to prevent this.
        # For simplicity, we rely on the prompt and the regex primarily.
        # If specific issues are seen, more targeted cleaning can be added here.
        # For example, removing thousands separators if they cause issues with float()
        cleaned_answer = answer.replace(',', '') # A common one
        try:
            # Try to convert the cleaned answer directly if it looks like a number
            if re.fullmatch(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cleaned_answer):
                return str(float(cleaned_answer)) # Convert to float then to string for consistent format
        except ValueError:
            pass # If still not a number

        return f"Could not extract a clear numerical answer. Raw model output: '{answer}'"
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
        if ' ' in cleaned_word: # Check for multiple words
            # Try to pick the first word if it seems like a sentence, or reject if it's clearly multi-word phrase
            first_word = cleaned_word.split(' ')[0]
            # Heuristic: if original input was short and became multi-word after cleaning, it might be intentional
            # but dictionary works best for single words.
            # For now, we'll stick to asking for single words.
            return f"The dictionary tool works best with single words. You provided: '{word}'. Try defining '{first_word}' or rephrase for a single word."
        
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{cleaned_word}"
        response = requests.get(api_url, timeout=10) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        
        if data and isinstance(data, list) and data[0].get('meanings'):
            definitions = []
            # Limit number of meanings and definitions to keep it concise
            for meaning in data[0]['meanings'][:2]: # Max 2 parts of speech
                part_of_speech = meaning.get('partOfSpeech', 'N/A')
                for def_obj in meaning.get('definitions', [])[:2]: # Max 2 definitions per part of speech
                    definition_text = def_obj.get('definition', 'No definition text.')
                    definitions.append(f"- ({part_of_speech}) {definition_text}")
            if definitions:
                return f"Definitions for '{word}':\n" + "\n".join(definitions)
            return f"No definitions found for '{word}' in the expected response structure."
        # Check for the specific "No Definitions Found" JSON response from this API
        elif isinstance(data, dict) and data.get("title") == "No Definitions Found":
             return f"Could not find a definition for '{word}' in the dictionary (API: No Definitions Found)."
        return f"No definition found for '{word}' or API response format unexpected. Raw API Response: {str(data)[:200]}" # Show a snippet of unexpected response
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return f"Could not find a definition for '{word}' in the dictionary (404 Not Found)."
        # Provide more detail from the response if available
        error_detail = ""
        try:
            error_content = http_err.response.json()
            error_detail = f" API Message: {error_content.get('title', '')} - {error_content.get('message', '')}"
        except ValueError: # If response is not JSON
            error_detail = f" Raw Response: {http_err.response.text[:100]}"
        return f"Dictionary API HTTP error: {http_err.response.status_code} for word '{word}'.{error_detail}"
    except requests.exceptions.RequestException as req_err: # Catches DNS errors, connection timeouts, etc.
        return f"Dictionary API request error: {req_err} for word '{word}'"
    except Exception as e: # Catch-all for other unexpected errors
        return f"An unexpected error occurred while fetching definition for '{word}': {str(e)}. Type: {type(e)}"

dictionary_tool = Tool.from_function(
    func=define_word,
    name="Dictionary",
    description="Use this tool when you need to find the definition of a specific word. Input should be the single word you want defined.",
    args_schema=DictionaryInput
)

tools = [calculator_tool, dictionary_tool]
if rag_tool:
    tools.insert(0, rag_tool) # RAG tool first if available

# Agent Callback Handler for streaming thoughts
class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.agent_logs = [] # Store logs here

    def on_agent_action(self, action, **kwargs) -> any:
        thought = action.log.strip().split("Action:")[0].strip() # Extract just the thought part
        tool_info = ""
        if action.tool and action.tool_input:
            # Sanitize tool input for display if it's a dict
            tool_input_str = str(action.tool_input)
            if isinstance(action.tool_input, dict):
                tool_input_str = ", ".join(f"{k}='{v}'" for k, v in action.tool_input.items())

            tool_info = f"🛠️ **Tool:** `{action.tool}` with input `{tool_input_str}`"
        
        log_message = f"🤔 **Thought:** {thought}\n{tool_info if tool_info else ''}"
        
        # Append to session state if available, otherwise to instance logs
        if hasattr(st, 'session_state') and "agent_logs" in st.session_state:
            st.session_state.agent_logs.append(log_message)
        else:
            self.agent_logs.append(log_message) # Fallback for non-Streamlit context if any

    def on_tool_end(self, output, name, **kwargs) -> any:
        # This log helps see the direct output of the tool
        log_message = f"✅ **Tool `{name}` Output:** `{str(output)[:300]}{'...' if len(str(output)) > 300 else ''}`"
        if hasattr(st, 'session_state') and "agent_logs" in st.session_state:
            st.session_state.agent_logs.append(log_message)
        else:
            self.agent_logs.append(log_message)

# Pull the prompt template for ReAct agent
# Using a more recent or robust chat-optimized ReAct prompt if available,
# but "hwchase17/react-chat" is a common starting point.
try:
    agent_prompt = hub.pull("hwchase17/react-chat")
except Exception as e:
    st.error(f"Could not pull react-chat prompt from Langchain Hub: {e}. Using a default fallback (may be less effective).")
    # Fallback prompt (simplified example, you might need a more robust one)
    from langchain.prompts import PromptTemplate
    # This is a very basic fallback and might not work well with all LLMs or complex scenarios
    AGENT_FALLBACK_PROMPT_TEMPLATE = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(AGENT_FALLBACK_PROMPT_TEMPLATE)


agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True for detailed console logs, good for debugging
    handle_parsing_errors="I encountered an issue processing the response or action. Please try rephrasing your request, or the tool might not have provided a usable output.", # Custom message for parsing errors
    max_iterations=7, # Limit iterations to prevent runaway agents
    # early_stopping_method="generate", # Optional: if LLM can signal completion
)

st.title(" RAG-Powered Multi-Agent Q&A Assistant (Groq & MiniLM)")
st.markdown("""
Ask questions about our documents (e.g., "Business hours?"), request calculations (e.g., "Calc 100/5 + 3"), or ask for word definitions (e.g., "Define ephemeral").
The agent will use Groq's fast LLMs and a lightweight local embedding model to respond.
**Note:** Data ingestion from the `./documents` folder happens on first load or if the vector store is missing/invalid.
""")

# Display the SQLite patch status message if not shown elsewhere prominently
if _sqlite_patch_applied_message and "main_ui_patch_message_shown" not in st.session_state:
    # Use a less intrusive way, like st.caption or a one-time toast
    # st.caption(_sqlite_patch_applied_message) # Alternative
    if not _patch_successful : # Only show prominently if there was an issue or warning
        st.toast(_sqlite_patch_applied_message, icon="⚠️" if "not found" in _sqlite_patch_applied_message else "❌")
    st.session_state.main_ui_patch_message_shown = True

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = [] # Initialize for the callback handler
if "retrieved_contexts" not in st.session_state:
    st.session_state.retrieved_contexts = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reset logs and contexts for the new query
    st.session_state.agent_logs = []
    st.session_state.retrieved_contexts = []
    
    agent_callback_handler = AgentCallbackHandler() # Instantiate handler for this run

    with st.chat_message("assistant"):
        thinking_message_placeholder = st.empty() # Placeholder for "Thinking..." or agent logs
        thinking_message_placeholder.markdown("🤖 Processing your request with Groq and MiniLM... Please wait...")
        
        final_answer_placeholder = st.empty() # Placeholder for the final answer
        
        try:
            response = agent_executor.invoke(
                {"input": prompt, "chat_history": st.session_state.messages[:-1]}, # Pass some history
                {"callbacks": [agent_callback_handler]}
            )
            answer = response.get('output', "No output found.")

            # Clear "Thinking..." and display agent logs if any
            thinking_message_placeholder.empty()

            # Display Agent Logs in an expander or directly
            with st.expander("Show Agent's Thought Process", expanded=False):
                if st.session_state.agent_logs:
                    for log_entry in st.session_state.agent_logs:
                        st.markdown(log_entry)
                        st.markdown("---") # Separator
                else:
                    st.markdown("*Agent may have answered directly, or detailed logs are in the console (verbose=True).*")
            
            if st.session_state.retrieved_contexts:
                with st.expander("Retrieved Context Snippets (if RAG tool was used)", expanded=False):
                    for i, context in enumerate(st.session_state.retrieved_contexts):
                        st.text_area(f"Context Snippet {i+1}", context, height=100, disabled=True)
            
            final_answer_placeholder.markdown(answer) # Display final answer
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            thinking_message_placeholder.empty() # Clear thinking message on error too
            error_message = f"An error occurred: {str(e)}"
            detailed_traceback = traceback.format_exc()
            st.error(error_message)
            # Optionally show full traceback in an expander for debugging
            with st.expander("Error Details (Traceback)"):
                st.code(detailed_traceback)
            
            final_answer_placeholder.markdown(f"Sorry, I encountered an error processing your request: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})