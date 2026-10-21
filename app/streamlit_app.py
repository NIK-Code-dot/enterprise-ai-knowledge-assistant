import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise AI Knowledge Assistant")
st.caption("Upload PDF documents and ask questions about their contents.")

with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload and index PDF", disabled=uploaded_file is None):
        try:
            with st.spinner("Uploading and indexing PDF..."):
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

            st.success(result.get("message", "Upload successful"))
            st.json(result)

        except requests.RequestException as error:
            st.error(f"Upload failed: {error}")

    st.divider()
    st.header("📚 Indexed documents")

    if st.button("Refresh documents"):
        st.session_state["refresh_documents"] = True

    try:
        response = requests.get(f"{API_URL}/documents", timeout=30)
        response.raise_for_status()
        documents = response.json().get("documents", [])

        if documents:
            for document in documents:
                st.write(f"• {document}")
        else:
            st.info("No indexed documents found.")

    except requests.RequestException:
        st.warning("Could not connect to the API.")

st.subheader("💬 Ask your documents")

question = st.text_area(
    "Enter your question",
    placeholder="What are the main points in my uploaded document?",
)

top_k = st.slider("Number of relevant text chunks", 1, 10, 4)

if st.button("Ask AI", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip(), "top_k": top_k},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

            st.markdown("### Answer")
            st.write(result.get("answer", "No answer returned."))

            sources = result.get("sources", [])
            if sources:
                st.markdown("### Sources")
                for source in sources:
                    with st.expander(
                        f"{source.get('source', 'Document')} — Page {source.get('page', '?')}"
                    ):
                        st.write(source.get("text", ""))

        except requests.RequestException as error:
            st.error(f"Question failed: {error}")
