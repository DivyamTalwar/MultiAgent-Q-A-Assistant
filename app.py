import streamlit as st

st.set_page_config(page_title="RAG Multi-Agent Q&A (Groq & MiniLM)", layout="wide")

import os
import requests
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
import traceback


load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in Streamlit secrets (GROQ_API_KEY) or as GROQ_API_KEY_LOCAL in your .env for local testing.")
    st.stop()

VECTOR_STORE_DIR = "vector_store"
LLM_MODEL_GROQ = "llama3-8b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

llm = ChatGroq(
    temperature=0,
    model_name=LLM_MODEL_GROQ,
    groq_api_key=GROQ_API_KEY
)

@st.cache_resource
def load_vector_store():
    if not os.path.exists(VECTOR_STORE_DIR) or not os.listdir(VECTOR_STORE_DIR):
        st.error(f"Vector store not found or empty at {VECTOR_STORE_DIR}. Please run ingest.py first.")
        return None
    
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_model)

vector_store = load_vector_store()
if vector_store is None:
    st.stop()

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

class AgentCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs) -> any:
        thought = action.log.strip()
        tool_info = ""
        if action.tool and action.tool_input:
            tool_info = f"➡️ Agent chose tool: **{action.tool}** with input: **{action.tool_input}**"
        log_message = f"🤔 Thought: {thought}\n{tool_info}"
        st.session_state.agent_logs.append(log_message)

    def on_tool_end(self, output, name, **kwargs) -> any:
        pass

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def run_rag_tool(query: str):
    result = rag_chain.invoke({"query": query})
    st.session_state.retrieved_contexts = [doc.page_content for doc in result['source_documents']]
    return result['result']

rag_tool = Tool(
    name="KnowledgeBaseSearch",
    func=run_rag_tool,
    description="Use this tool to answer questions about company policies, product specifications, FAQs, and general knowledge found in the provided documents. Input should be a complete question.",
)

class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate. E.g., '2 + 2', 'sqrt(25) * 3'")

def calculate_expression(expression: str):
    try:
        calculator_prompt = f"Evaluate the following mathematical expression and return only the final numerical result (no explanation, just the number itself, ensure it is a valid number): {expression}"
        response = llm.invoke(calculator_prompt)
        answer = response.content.strip()
        match = re.search(r'-?\d+\.?\d*', answer)
        if match:
            return match.group(0)
        parts = answer.split()
        for part in reversed(parts):
            try:
                num_part = part.replace(',', '').replace('$', '').replace('€', '').strip()
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
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{cleaned_word}")
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and data[0].get('meanings'):
            definitions = []
            for meaning in data[0]['meanings'][:2]:
                for definition_obj in meaning.get('definitions', [])[:2]:
                    definitions.append(f"- ({meaning['partOfSpeech']}) {definition_obj['definition']}")
            return "\n".join(definitions) if definitions else f"No definitions found for '{word}'."
        return f"No definition found for '{word}' or API response format unexpected."
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return f"Could not find a definition for '{word}' in the dictionary."
        return f"Dictionary API HTTP error: {http_err}"
    except Exception as e:
        return f"Error fetching definition for '{word}': {str(e)}"

dictionary_tool = Tool.from_function(
    func=define_word,
    name="Dictionary",
    description="Use this tool when you need to find the definition of a specific word. Input should be the single word you want defined.",
    args_schema=DictionaryInput
)

tools = [rag_tool, calculator_tool, dictionary_tool]
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

                st.subheader("Agent Decision Path & Thoughts:")
                if not st.session_state.agent_logs:
                    st.markdown("*No specific tool actions logged, or agent answered directly. Check console for verbose logs.*")
                for log_entry in st.session_state.agent_logs:
                    st.markdown(f"{log_entry}")
                
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
                st.error(error_message)
                st.error(f"Traceback: {traceback.format_exc()}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})