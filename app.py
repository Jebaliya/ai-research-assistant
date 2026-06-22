import os
import streamlit as st
from dotenv import load_dotenv
from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.vector_store import ChromaVectorStore
from rag.retriever import Retriever
from rag.llm import GeminiLLM
#
load_dotenv()

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []


def add_to_memory(user_msg, assistant_msg):
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    st.session_state.chat_history = st.session_state.chat_history[-20:]


def get_history_as_text():
    return "\n".join([f"{m['role'].upper()}: {m['content']}"
                      for m in st.session_state.chat_history])


# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    
    layout="wide"
)

# Custom CSS - ChatGPT style
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #F7F7F8;
        color: #202123;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5E5;
    }

    [data-testid="stSidebar"] * {
        color: #202123 !important;
    }

    .user-msg {
        background-color: #F7F7F8;
        padding: 20px 40px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
        border-bottom: 1px solid #E5E5E5;
    }

    .assistant-msg {
        background-color: #FFFFFF;
        padding: 20px 40px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
        border-bottom: 1px solid #E5E5E5;
    }

    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }

    .user-avatar {
        background-color: #5436DA;
    }

    .bot-avatar {
        background-color: #11A37F;
    }

    .msg-content {
        flex: 1;
        line-height: 1.7;
        font-size: 15px;
        color: #202123;
        padding-top: 4px;
    }

    .chat-area {
        margin-bottom: 120px;
        margin-top: 10px;
    }

    .source-badge {
        display: inline-block;
        background-color: #F0F0F0;
        border: 1px solid #E5E5E5;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        color: #666;
        margin: 4px 4px 0 0;
    }

    .file-item {
        background-color: #F0FFF8;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 4px 0;
        font-size: 13px;
        color: #202123;
        border-left: 3px solid #11A37F;
    }

    .main-title {
        text-align: center;
        color: #202123;
        font-size: 28px;
        font-weight: 600;
        padding: 40px 0 20px;
    }

    .stButton button {
        background-color: #11A37F;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #0D8F6F;
    }

    .stTextInput input {
        background-color: #FFFFFF;
        color: #202123;
        border: 1px solid #D9D9E3;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
vector_store = ChromaVectorStore()
retriever = Retriever(vector_store)

try:
    llm = GeminiLLM()
except Exception as e:
    llm = None

# Sidebar
with st.sidebar:
    st.markdown("## AI Research Assistant")
    st.markdown("---")

    # Multiple file upload
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files_list:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        text = load_document(file_bytes, uploaded_file.name)
                        chunks = chunk_text(text)
                        documents = [
                            {
                                "id": f"{uploaded_file.name}-{i}",
                                "text": chunk,
                                "metadata": {"source": uploaded_file.name}
                            }
                            for i, chunk in enumerate(chunks)
                        ]
                        vector_store.add_documents(documents)
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")

    # Show uploaded files
    if st.session_state.uploaded_files_list:
        st.markdown("### 📚 Indexed Documents")
        for fname in st.session_state.uploaded_files_list:
            st.markdown(f'<div class="file-item">📄 {fname}</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if llm is None:
        st.error("⚠️ API key missing in .env file")

# Main chat area
st.markdown('<div class="main-title">AI Research Assistant</div>', unsafe_allow_html=True)

# Welcome message
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center; color:#8E8EA0; padding: 60px 20px;">
        <div style="font-size:48px;"></div>
        <div style="font-size:18px; margin-top:16px;">Upload documents from the sidebar and start asking questions!</div>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="avatar user-avatar">👤</div>
            <div class="msg-content">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-msg">
            <div class="avatar bot-avatar"></div>
            <div class="msg-content">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input
st.markdown("---")
col1, col2 = st.columns([6, 1])

with col1:
    question = st.text_input(
        "Message",
        placeholder="Ask a question about your documents...",
        label_visibility="collapsed",
        key="question_input"
    )

with col2:
    submit = st.button("Send ➤")

if submit and question:
    if llm is None:
        st.error("Gemini/Groq not available. Check API key.")
    elif not st.session_state.uploaded_files_list:
        st.warning("Please upload at least one document first.")
    else:
        with st.spinner("Thinking..."):
            results = retriever.retrieve(question, top_k=3)
            if not results:
                st.warning("No relevant content found.")
            else:
                answer = llm.generate_answer(
                    question, results, get_history_as_text()
                )
                add_to_memory(question, answer)
                st.rerun()