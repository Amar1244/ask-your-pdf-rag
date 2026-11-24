# Ask Your PDF: RAG Chatbot

A chatbot that lets you upload any PDF and ask questions about it in plain English. Answers come from the document's own content, and the chat remembers earlier questions so you can ask follow-ups.

## What it does

- **Chat with any PDF.** Upload a document and ask questions about it. The answers are grounded in the text of the file.
- **Retrieval-Augmented Generation (RAG).** The PDF is split into overlapping chunks, converted to embeddings, and stored in a vector database. For each question, the most relevant chunks are retrieved and passed to the LLM.
- **Conversation memory.** Follow-up questions work because the chat history is kept in context.
- **Saved index.** The vector store is persisted to disk, so a processed document is reused instead of being re-indexed.
- **Simple web UI.** A Streamlit app handles upload, questions and chat history.

## Keywords

Generative AI · LLM · RAG · Retrieval-Augmented Generation · Document Q&A · Chatbot · Vector Database · Embeddings · LangChain · Google Gemini · ChromaDB · Streamlit · Python

## Tech Stack

- Python
- [LangChain](https://www.langchain.com/) for the retrieval and conversation chain
- [Google Gemini](https://ai.google.dev/) (`gemini-2.0-flash`) for answers and Gemini embeddings for search
- [ChromaDB](https://www.trychroma.com/) as the vector store
- PyMuPDF for PDF loading
- [Streamlit](https://streamlit.io/) for the UI

## Project Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit app: upload, questions and chat history |
| `main.py` | RAG pipeline: PDF loading and chunking, embeddings, vector store and the conversational chain |
| `requirements.txt` | Python dependencies |

## Author

**Amar Chaurasiya**, Data Scientist working as an AI Engineer. Open to full-time roles and freelance work.

- Email: [amarchaurasiya124@gmail.com](mailto:amarchaurasiya124@gmail.com)
- LinkedIn: [linkedin.com/in/amar-chaurasiya-163788262](https://www.linkedin.com/in/amar-chaurasiya-163788262)
- Portfolio: [amarchaurasiya.vercel.app](https://amarchaurasiya.vercel.app)
