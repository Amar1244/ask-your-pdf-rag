# app.py

import streamlit as st
import tempfile
import os

from main import load_and_split_pdf, create_vectorstore, build_qa_chain
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Page configuration
st.set_page_config(page_title="AskYourPDF AI", page_icon="📄")
st.title("📄 AskYourPDF AI")
st.caption("Built with LangChain + Gemini + ChromaDB (RAG system)")

# Initialize session state
if "qa_chain" not in st.session_state:
    # Try loading vectorstore if it exists
    if os.path.exists("vectordb"):
        with st.spinner("🔄 Loading saved vector database..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma(
                persist_directory="vectordb",
                embedding_function=embeddings
            )
            st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.success("✅ Loaded existing PDF context!")
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("📄 Reading and indexing PDF..."):
        chunks = load_and_split_pdf(pdf_path)
        vectorstore = create_vectorstore(chunks)
        qa_chain = build_qa_chain(vectorstore)
        st.session_state.qa_chain = qa_chain
        st.session_state.chat_history = []  # Reset chat history on new upload
        st.success("✅ PDF is ready! Ask your questions below 👇")

# Input box for question
query = st.text_input("💬 Ask a question about the PDF")

# Process query
if query and st.session_state.qa_chain:
    with st.spinner("🧠 Thinking..."):
        response = st.session_state.qa_chain.run(query)
        st.session_state.chat_history.append((query, response))

# Display chat history
if st.session_state.chat_history:
    st.subheader("🧠 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.markdown("---")
