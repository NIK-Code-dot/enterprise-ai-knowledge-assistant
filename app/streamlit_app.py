
import os
import requests
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).strip().rstrip("/")

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None


# --------------------------------------------------
# API helpers
# --------------------------------------------------
def api_error_message(error):
    response = getattr(error, "response", None)

    if response is not None:
        try:
            detail = response.json()
            if isinstance(detail, dict):
                return str(
                    detail.get("detail")
                    or detail.get("message")
                    or detail
                )
            return str(detail)
        except (ValueError, requests.RequestException):
            return response.text or str(error)

    return str(error)


def get_documents():
    try:
        response = requests.get(
            f"{API_URL}/documents",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ("documents", "items", "data"):
                if isinstance(data.get(key), list):
                    return data[key]

        return []

    except requests.RequestException as error:
        st.warning(
            f"Could not load documents: "
            f"{api_error_message(error)}"
        )
        return None


def upload_document(uploaded_file):
    if uploaded_file is None:
        st.warning("Please select a PDF file first.")
        return

    if not uploaded_file.name.lower().endswith(".pdf"):
        st.error("Only PDF files are supported.")
        return

    try:
        with st.spinner(
            "Uploading and indexing PDF. Please wait..."
        ):
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

            try:
                result = response.json()
            except ValueError:
                result = {
                    "message": response.text
                    or "PDF uploaded successfully."
                }

        st.session_state.last_upload = {
            "name": uploaded_file.name,
            "result": result,
        }
        st.session_state.messages = (
            st.session_state.messages
        )

        message = (
            result.get("message", "PDF uploaded successfully.")
            if isinstance(result, dict)
            else "PDF uploaded successfully."
        )

        st.success(message)
        st.rerun()

    except requests.RequestException as error:
        st.error(
            f"PDF upload failed: {api_error_message(error)}"
        )


def ask_question(question, top_k, document_id=None):
    payload = {
        "question": question,
        "top_k": top_k,
    }

    if document_id:
        payload["document_id"] = document_id

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            return str(data), []

        answer = (
            data.get("answer")
            or data.get("response")
            or data.get("result")
            or "No answer was returned by the API."
        )

        sources = data.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        return str(answer), sources

    except requests.RequestException as error:
        return (
            f"API request failed: {api_error_message(error)}",
            [],
        )


def document_name(document):
    if isinstance(document, str):
        return document

    if isinstance(document, dict):
        return str(
            document.get("filename")
            or document.get("name")
            or document.get("source")
            or document.get("document_id")
            or document.get("id")
            or "Unnamed document"
        )

    return str(document)


# --------------------------------------------------
# Theme
# --------------------------------------------------
with st.sidebar:
    st.title("📚 Enterprise AI")
    st.caption("Knowledge Assistant")

    st.selectbox(
        "Appearance",
        ["Light", "Dark"],
        key="theme",
    )

if st.session_state.theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #111827;
            color: #f3f4f6;
        }
        [data-testid="stSidebar"] {
            background-color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Sidebar: PDF upload and documents
# --------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("📤 Upload a PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="sidebar_pdf",
        help="Select a PDF to upload and index.",
    )

    if uploaded_file is not None:
        st.caption(
            f"Selected: {uploaded_file.name}"
        )
        st.caption(
            f"Size: {uploaded_file.size / 1024:.1f} KB"
        )

    # No form: button click executes immediately.
    if st.button(
        "Upload and index",
        type="primary",
        key="upload_pdf_button",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        upload_document(uploaded_file)

    if st.session_state.last_upload:
        last = st.session_state.last_upload
        st.caption(
            f"Last uploaded: {last['name']}"
        )

    st.divider()
    st.subheader("📁 Indexed documents")

    if st.button(
        "↻ Refresh documents",
        use_container_width=True,
    ):
        st.rerun()

    documents = get_documents()

    if documents is None:
        st.caption("Document list unavailable.")
    elif documents:
        for document in documents:
            st.write(f"📄 {document_name(document)}")
    else:
        st.info("No documents found yet.")


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------
st.title("Enterprise AI Knowledge Assistant")
st.write(
    "Upload your documents and ask questions "
    "using AI-powered retrieval."
)

tab_dashboard, tab_chat, tab_documents = st.tabs(
    ["🏠 Overview", "💬 Ask AI", "📁 Documents"]
)

with tab_dashboard:
    col1, col2, col3 = st.columns(3)

    document_count = len(documents or [])

    col1.metric("Indexed documents", document_count)
    col2.metric(
        "Questions in this session",
        sum(
            1
            for message in st.session_state.messages
            if message["role"] == "user"
        ),
    )
    col3.metric(
        "API endpoint",
        "Configured" if API_URL else "Missing",
    )

    st.subheader("Getting started")
    st.markdown(
        """
        1. Upload a PDF from the sidebar.
        2. Wait for indexing to finish.
        3. Open **Ask AI** and enter your question.
        4. Review the answer and any retrieved sources.
        """
    )

with tab_chat:
    st.subheader("Ask questions about your documents")

    top_k = st.slider(
        "Number of retrieved passages",
        min_value=1,
        max_value=10,
        value=4,
    )

    document_options = ["All documents"] + [
        document_name(doc)
        for doc in (documents or [])
    ]

    selected_document = st.selectbox(
        "Search scope",
        document_options,
    )

    # Display previous messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        if isinstance(source, dict):
                            st.markdown(
                                f"**Source:** "
                                f"{source.get('source', 'Unknown')}"
                            )

                            if source.get("page") is not None:
                                st.caption(
                                    f"Page: {source['page']}"
                                )

                            if source.get("text"):
                                st.write(source["text"])
                        else:
                            st.write(str(source))

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # The backend may expect an actual document ID.
        # Send a filename only when the API uses that convention.
        selected_id = None

        if selected_document != "All documents":
            for doc in (documents or []):
                if document_name(doc) == selected_document:
                    if isinstance(doc, dict):
                        selected_id = (
                            doc.get("document_id")
                            or doc.get("id")
                            or doc.get("filename")
                            or doc.get("name")
                        )
                    else:
                        selected_id = doc
                    break

        with st.chat_message("assistant"):
            with st.spinner("Searching your knowledge base..."):
                answer, sources = ask_question(
                    question,
                    top_k,
                    selected_id,
                )

            st.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        if isinstance(source, dict):
                            st.markdown(
                                f"**Source:** "
                                f"{source.get('source', 'Unknown')}"
                            )
                            if source.get("page") is not None:
                                st.caption(
                                    f"Page: {source['page']}"
                                )
                            if source.get("text"):
                                st.write(source["text"])
                        else:
                            st.write(str(source))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

with tab_documents:
    st.subheader("Your documents")

    if st.button("Refresh document list"):
        st.rerun()

    if documents is None:
        st.warning(
            "Unable to retrieve the document list from the API."
        )
    elif not documents:
        st.info(
            "No indexed documents are currently listed."
        )
    else:
        for index, document in enumerate(documents, start=1):
            with st.container(border=True):
                st.write(
                    f"**{index}. {document_name(document)}**"
                )

                if isinstance(document, dict):
                    for key in ("page_count", "pages", "chunks"):
                        if document.get(key) is not None:
                            st.caption(
                                f"{key.replace('_', ' ').title()}: "
                                f"{document[key]}"
                            )
