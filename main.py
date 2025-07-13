# 📄 rag_pipeline.py – With Detailed Comments

import os  # Used to interact with the operating system (for reading environment variables)
from dotenv import load_dotenv  # Loads environment variables from a .env file

# Load PDF documents using PyMuPDF (high accuracy and fast)
from langchain_community.document_loaders import PyMuPDFLoader

# Splits large text into manageable overlapping chunks for vector embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imports Gemini chat model and embedding model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Vector database to store and search over embeddings
from langchain.vectorstores import Chroma

# Chain that connects LLM + retriever + memory into a conversational QA system
from langchain.chains import ConversationalRetrievalChain

# Memory component that stores chat history across user questions
from langchain.memory import ConversationBufferMemory


#  Step 0: Load the Gemini API Key securely from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key is None:
    raise ValueError("❌ GEMINI_API_KEY is not set in the .env file")

os.environ["GOOGLE_API_KEY"] = gemini_api_key


# Step 1: Load PDF and split into text chunks
def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)   # Load PDF file
    docs = loader.load()                # Read pages into a document list

    # Split pages into smaller overlapping text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)  # Apply splitting

    return chunks  # Return list of small text chunks


#  Step 2: Convert chunks into vector embeddings and store in Chroma DB
def create_vectorstore(chunks):
    # Embedding model that converts text to vectors (numeric representations)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create Chroma vector DB and persist it to disk for reuse
    vectorstore = Chroma.from_documents(
        documents=chunks,                # Chunks of text to embed
        embedding=embeddings,            # Embedding function/model
        persist_directory="vectordb"     # Folder where data is saved
    )

    vectorstore.persist()  # Save to disk so we don’t need to reprocess next time
    return vectorstore     # Return the retrievable vector store


#  Step 3: Build the Conversational QA (RAG) chain
def build_qa_chain(vectorstore):
    # Load Gemini LLM for chat-based responses
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    # Enable memory to track chat history across questions
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key used to track conversation
        return_messages=True        # Return full messages, not just strings
    )

    # Create the conversational retrieval chain (RAG)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,                               # The LLM for answering
        retriever=vectorstore.as_retriever(), # Vector search for relevant chunks
        memory=memory                         # Store and recall chat history
    )

    return qa_chain  # Return the full RAG pipeline

""" Summary:

load_and_split_pdf(file_path)
    → Loads a PDF and splits it into small, overlapping chunks for better retrieval.

create_vectorstore(chunks)
    → Converts each chunk into vector form using Gemini embeddings and saves them into a Chroma vector database.

build_qa_chain(vectorstore)
    → Builds a LangChain RAG pipeline: Retriever + Gemini LLM + Memory for real-time document Q&A.
"""
