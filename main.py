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

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            print(response, show_source=True)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) 
