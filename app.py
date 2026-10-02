import streamlit as st
import PyPDF2
import io
from groq import Groq
import re
from collections import Counter
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f6ff;
        border: 1px solid #c8dcff;
        margin-bottom: 15px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #effaf2;
        border: 1px solid #b9e6c1;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff8e6;
        border: 1px solid #f2d48a;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = {
        "filename": "",
        "pages": 0,
        "words": 0,
        "chunks": 0
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = ""


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# Groq model
DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):
    """
    Extract text from PDF file.
    """

    pdf_reader = PyPDF2.PdfReader(
        pdf_file
    )

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += (
                page_text
                + "\n"
            )

    return (
        text,
        len(pdf_reader.pages)
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):
    """
    Split text into chunks with overlap.
    """

    chunks = []

    start = 0

    text_len = len(text)

    while start < text_len:

        end = start + chunk_size

        chunk = text[
            start:end
        ]

        # Try to break at sentence boundary
        if end < text_len:

            last_period = (
                chunk.rfind(".")
            )

            if (
                last_period
                > chunk_size * 0.5
            ):

                end = (
                    start
                    + last_period
                    + 1
                )

                chunk = text[
                    start:end
                ]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start = end - overlap

    return chunks


# ============================================================
# TOKENIZATION
# ============================================================

def simple_tokenize(text):
    """
    Simple tokenization.
    """

    words = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    return words


# ============================================================
# SIMILARITY CALCULATION
# ============================================================

def calculate_similarity(
    query,
    chunk
):
    """
    Calculate simple word overlap
    similarity between query and chunk.
    """

    query_words = set(
        simple_tokenize(query)
    )

    chunk_words = simple_tokenize(
        chunk
    )

    chunk_word_set = set(
        chunk_words
    )

    # Calculate overlap
    overlap = (
        query_words.intersection(
            chunk_word_set
        )
    )

    if not overlap:

        return 0

    # Calculate TF-IDF-like score
    chunk_counter = Counter(
        chunk_words
    )

    score = sum(
        chunk_counter[word]
        for word in overlap
    )

    # Normalize by chunk length
    return (
        score / len(chunk_words)
        if chunk_words
        else 0
    )


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    query,
    top_k=3
):
    """
    Retrieve the most relevant PDF chunks.
    """

    pdf_chunks = (
        st.session_state.pdf_chunks
    )

    if not pdf_chunks:

        return []

    chunk_scores = []

    for i, chunk in enumerate(
        pdf_chunks
    ):

        score = calculate_similarity(
            query,
            chunk
        )

        chunk_scores.append(
            (
                i,
                score,
                chunk
            )
        )

    # Sort by score
    chunk_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Return chunks with non-zero scores
    relevant = [
        chunk
        for _, score, chunk
        in chunk_scores[:top_k]
        if score > 0
    ]

    return relevant


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer_with_groq(
    query,
    context,
    api_key
):
    """
    Generate answer using Groq API.
    """

    try:

        if not api_key:

            return (
                "❌ Groq API key is missing.\n\n"
                "Please enter your Groq API key "
                "in the sidebar."
            )

        # Create Groq client
        groq_client = Groq(
            api_key=api_key
        )

        # Prepare prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the user's question based ONLY
on the provided context from the PDF.

Context from PDF:
{context}

Question:
{query}

Instructions:
- Answer based only on the provided context.
- Do not invent information.
- If the context does not contain the answer,
  clearly say that the information was not found
  in the provided PDF context.
- Be concise and accurate.
- Explain the answer clearly.
- Cite specific parts of the context when possible.

Answer:
"""

        # Call Groq
        chat_completion = (
            groq_client
            .chat
            .completions
            .create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI "
                            "assistant that answers "
                            "questions using PDF "
                            "documents."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=1024
            )
        )

        return (
            chat_completion
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        error_msg = str(e)

        # Invalid API key
        if (
            "invalid_api_key"
            in error_msg.lower()
        ):

            return (
                "❌ Invalid Groq API Key.\n\n"
                "Please check your API key "
                "and try again."
            )

        # Rate limit
        elif (
            "rate_limit"
            in error_msg.lower()
        ):

            return (
                "❌ Groq rate limit exceeded.\n\n"
                "Please wait a moment and try again."
            )

        # Model error
        elif (
            "model_not_found"
            in error_msg.lower()
        ):

            return (
                "❌ The model "
                f"`{DEFAULT_MODEL}` "
                "is unavailable for this API "
                "account."
            )

        # Network / 403
        elif (
            "403"
            in error_msg
        ):

            return (
                "❌ Groq returned HTTP 403.\n\n"
                "Access was denied. Please check "
                "your network connection, VPN, "
                "proxy, or Groq access."
            )

        # Other errors
        else:

            return (
                "❌ Groq API Error:\n\n"
                f"{error_msg}"
            )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):
    """
    Process uploaded PDF.
    """

    try:

        # Read uploaded PDF
        pdf_bytes = (
            uploaded_file.getvalue()
        )

        pdf_file = io.BytesIO(
            pdf_bytes
        )

        # Extract text
        pdf_text, pages = (
            extract_text_from_pdf(
                pdf_file
            )
        )

        if not pdf_text.strip():

            return (
                False,
                "❌ No text found in PDF."
            )

        # Create chunks
        chunks = chunk_text(
            pdf_text
        )

        # Count words
        word_count = len(
            pdf_text.split()
        )

        # Save PDF text
        st.session_state.pdf_text = (
            pdf_text
        )

        # Save chunks
        st.session_state.pdf_chunks = (
            chunks
        )

        # Save information
        st.session_state.pdf_info = {

            "filename":
                uploaded_file.name,

            "pages":
                pages,

            "words":
                word_count,

            "chunks":
                len(chunks)
        }

        # Save processed filename
        st.session_state.processed_file_name = (
            uploaded_file.name
        )

        # Clear previous chat
        st.session_state.chat_history = []

        return (
            True,
            f'✅ PDF "{uploaded_file.name}" '
            "processed successfully!"
        )

    except Exception as e:

        return (
            False,
            f"❌ PDF processing error: {str(e)}"
        )


# ============================================================
# CLEAR EVERYTHING
# ============================================================

def clear_all():

    st.session_state.pdf_chunks = []

    st.session_state.pdf_text = ""

    st.session_state.pdf_info = {

        "filename": "",
        "pages": 0,
        "words": 0,
        "chunks": 0
    }

    st.session_state.chat_history = []

    st.session_state.processed_file_name = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown("---")

    # --------------------------------------------------------
    # GROQ API KEY
    # --------------------------------------------------------

    st.subheader("🔑 Groq API Key")

    groq_api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Enter your Groq API key here."
    )

    if groq_api_key:

        st.success(
            "✅ API key entered"
        )

    else:

        st.warning(
            "⚠️ API key required"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload a text-based PDF."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CLEAR BUTTON
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat & PDF",
        use_container_width=True
    ):

        clear_all()

        st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.subheader("🤖 AI Model")

    st.info(
        DEFAULT_MODEL
    )

    st.markdown("---")

    st.caption(
        "📚 PDF RAG Chatbot"
    )

    st.caption(
        "Powered by Streamlit + PyPDF2 + Groq"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📚 PDF RAG Chatbot
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions about your PDF using AI
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    # Process only a new PDF
    if (
        st.session_state.processed_file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Processing PDF..."
        ):

            success, message = (
                process_pdf(
                    uploaded_file
                )
            )

        if success:

            st.success(
                message
            )

        else:

            st.error(
                message
            )

    elif (
        st.session_state.pdf_chunks
    ):

        st.success(
            f'📄 "{uploaded_file.name}" '
            "is ready for questions."
        )


# ============================================================
# PDF INFORMATION
# ============================================================

if st.session_state.pdf_chunks:

    pdf_info = (
        st.session_state.pdf_info
    )

    st.markdown(
        '<div class="info-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
### 📄 Current Document

**File:** `{pdf_info["filename"]}`

**Pages:** `{pdf_info["pages"]}`

**Words:** `{pdf_info["words"]}`

**Chunks:** `{pdf_info["chunks"]}`
        """
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# API KEY WARNING
# ============================================================

if not groq_api_key:

    st.warning(
        "🔑 Enter your Groq API key in the "
        "sidebar to ask questions."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.chat_history
):

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "💬 Ask a question about your PDF..."
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not query.strip():

        st.warning(
            "❌ Please enter a question."
        )

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    elif not st.session_state.pdf_chunks:

        st.error(
            "❌ Please upload a PDF document first."
        )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    elif not groq_api_key:

        st.error(
            "❌ Please enter your Groq API key "
            "in the sidebar."
        )

    else:

        # ----------------------------------------------------
        # DISPLAY USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                query
            )

        # Save user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        # ----------------------------------------------------
        # RETRIEVE RELEVANT CHUNKS
        # ----------------------------------------------------

        relevant_chunks = (
            retrieve_relevant_chunks(
                query,
                top_k=3
            )
        )

        # ----------------------------------------------------
        # NO RELEVANT CHUNKS
        # ----------------------------------------------------

        if not relevant_chunks:

            answer = (
                "❌ No relevant information "
                "was found in the PDF for "
                "your question.\n\n"
                "Try rephrasing your question "
                "or asking something else "
                "about the document."
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        else:

            # Combine chunks
            context = (
                "\n\n---\n\n"
                .join(
                    relevant_chunks
                )
            )

            # Generate answer
            with st.spinner(
                "🤖 Thinking..."
            ):

                answer = (
                    generate_answer_with_groq(
                        query,
                        context,
                        groq_api_key
                    )
                )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

        # Save assistant message
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# EMPTY STATE
# ============================================================

if (
    not st.session_state.pdf_chunks
    and not st.session_state.chat_history
):

    st.markdown(
        """
        <div class="warning-box">

### 👋 Welcome!

1. Upload a PDF from the sidebar.
2. Enter your Groq API key.
3. Wait for the PDF to be processed.
4. Ask questions using the chat box below.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
### ✨ Features

- 📄 PDF text extraction
- ✂️ Text chunking with overlap
- 🔎 Keyword-based retrieval
- 🤖 Groq `openai/gpt-oss-120b`
- 💬 Chat interface
- 📊 PDF statistics
- 🧹 Clear/reset functionality
        """
    )