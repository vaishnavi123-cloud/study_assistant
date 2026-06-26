import streamlit as st
import os
import shutil
from modules.pdf_loader import load_pdf_text
from modules.text_splitter import split_text
from modules.vector_store import (
    create_vector_store,
    load_vector_store
)
from modules.rag_chain import get_llm

st.set_page_config(
    page_title="AI Study Assistant",
    layout="wide"
)
st.title("📚 AI Study Assistant using RAG + Ollama")


def _load_llm_settings_from_secrets():
    # Streamlit Cloud secrets are not always available as process env vars.
    for key in [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ]:
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


def _extract_response_text(response):
    if hasattr(response, "content"):
        return response.content
    return str(response)


_load_llm_settings_from_secrets()

# Check if vector store exists
vector_db_path = "vector_db"
metadata_file = "document_metadata.txt"
if os.path.exists(vector_db_path) and os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        doc_name = f.read().strip()
    st.success(f"📄 Document about vector algebra is already loaded and processed!")
    pdf_loaded = True
    if st.button("Upload a new document"):
        shutil.rmtree(vector_db_path)
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        st.rerun()
else:
    pdf_loaded = False

if not pdf_loaded:
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type="pdf",
        key="pdf_uploader"
    )
    if uploaded_pdf:
        with open(uploaded_pdf.name, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        with st.spinner("Reading PDF..."):
            raw_text = load_pdf_text(uploaded_pdf.name)
        with st.spinner("Splitting text..."):
            chunks = split_text(raw_text)
        with st.spinner("Creating vector DB..."):
            create_vector_store(chunks)

        # Save metadata
        with open(metadata_file, "w") as f:
            f.write(uploaded_pdf.name)

        st.success("PDF processed successfully!")
        pdf_loaded = True
        st.rerun()  # Refresh to show loaded state

if pdf_loaded:
    question = st.text_input(
        "Ask a question from your PDF"
    )
    if question:
        try:
            db = load_vector_store()
            docs = db.similarity_search(
                question,
                k=2
            )
            llm, provider = get_llm()
        except Exception as exc:
            st.error(
                "Unable to initialize retrieval/LLM. "
                "Configure a reachable OLLAMA_BASE_URL and existing OLLAMA_MODEL."
            )
            st.info(
                "If you are using Streamlit Cloud, add these in App Settings > Secrets:\n"
                "OLLAMA_BASE_URL=https://your-ollama-host\n"
                "OLLAMA_MODEL=phi3"
            )
            st.exception(exc)
            st.stop()

        st.caption(f"Provider: {provider}")

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""

        try:
            response = llm.invoke(prompt)
        except Exception as exc:
            st.error(
                "The model request failed. Check your provider configuration "
                "(OLLAMA_BASE_URL/OLLAMA_MODEL)."
            )
            st.exception(exc)
            st.stop()

        st.subheader("Answer")
        st.write(_extract_response_text(response))
