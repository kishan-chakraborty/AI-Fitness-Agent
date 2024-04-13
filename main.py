"""
Implementing a simple chatbot on diet and health using RAG with llamaindex and
deploting the bot in streamlit.
"""
import os
import os.path
import streamlit as st

from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader,
                              load_index_from_storage, ServiceContext,
                              StorageContext)
from llama_index.llms.openai import OpenAI

load_dotenv()
STORAGE_PATH = "./vectorstore"
DOCUMENTS_PATH = "./documents"

llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

@st.cache_resource(show_spinner=False)
def initialize():
    """
        Initialize the RAG process.
    """
    if not os.path.exists(STORAGE_PATH):
        # Ingest documents
        documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()
        # Creating vector index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_PATH)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
        index = load_index_from_storage(storage_context)
    return index

index = initialize()

st.title("Ask the Document")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]
