import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.vector_store import ChromaVectorStore
from rag.retriever import Retriever
from rag.llm import GeminiLLM
from rag.agent import ResearchAgent
from rag.evaluator import Evaluator
from rag.logger import trace_query, trace_document_upload
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm
from reportlab.lib import colors


load_dotenv()

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-primary: #0A0A0B;
        --bg-secondary: #111113;
        --bg-tertiary: #17171A;
        --bg-elevated: #1C1C20;
        --border-subtle: #26262B;
        --border-strong: #33333A;
        --text-primary: #EDEDEF;
        --text-secondary: #A1A1AA;
        --text-tertiary: #71717A;
        --accent: #6366F1;
        --accent-hover: #7C7FF2;
        --accent-soft: rgba(99, 102, 241, 0.12);
        --success: #22C55E;
        --success-soft: rgba(34, 197, 94, 0.10);
        --warning: #F59E0B;
        --warning-soft: rgba(245, 158, 11, 0.10);
        --danger: #EF4444;
        --danger-soft: rgba(239, 68, 68, 0.10);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.35);
        --shadow-lg: 0 12px 32px rgba(0,0,0,0.45);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    * { box-sizing: border-box; }

    html, body, .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #45454D; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary); }
    [data-testid="stSidebar"] h3 {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
        margin: 4px 0 10px !important;
    }

    .sidebar-section {
        padding: 2px 0 14px;
    }

    /* Brand block */
    .brand-block {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 2px 18px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 16px;
    }
    .brand-mark {
        width: 34px; height: 34px;
        background: var(--accent-soft);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: var(--radius-sm);
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700; color: var(--accent);
        letter-spacing: -0.02em;
        flex-shrink: 0;
    }
    .brand-name {
        font-size: 14px; font-weight: 600; color: var(--text-primary);
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 11px; color: var(--text-tertiary); margin-top: 1px;
    }

    /* User strip */
    .user-strip {
        display: flex; align-items: center; gap: 10px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 9px 12px;
        margin-bottom: 4px;
    }
    .user-strip-avatar {
        width: 26px; height: 26px; border-radius: 50%;
        background: var(--accent);
        color: #fff; font-size: 11px; font-weight: 600;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .user-strip-name {
        font-size: 13px; font-weight: 500; color: var(--text-primary);
    }
    .user-strip-role {
        font-size: 11px; color: var(--text-tertiary);
    }

    /* File uploader */
    [data-testid="stFileUploader"] > div {
        border: 1px dashed var(--border-strong) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-tertiary) !important;
        padding: 14px !important;
        transition: border-color .15s ease;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: var(--accent) !important;
    }
    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-secondary) !important;
    }

    /* Indexed file chip */
    .file-item {
        display: flex; align-items: center; gap: 8px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 7px 10px;
        margin: 4px 0;
        font-size: 12.5px;
        color: var(--text-secondary);
    }
    .file-item::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--success);
        flex-shrink: 0;
    }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: 14px 0 !important;
    }

    /* ── Fixed header ── */
    .fixed-header {
        position: fixed;
        top: 0; left: 0; right: 0;
        background: rgba(10, 10, 11, 0.85);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-subtle);
        z-index: 999;
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .fixed-header-inner {
        max-width: 760px;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .fixed-header-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .fixed-header-sub {
        font-size: 12px;
        color: var(--text-tertiary);
        margin-top: 2px;
        text-align: center;
    }

    /* ── Chat area ── */
    .chat-wrapper {
        margin-top: 92px;
        margin-bottom: 110px;
        padding: 0 16px;
        max-width: 760px;
        margin-left: auto;
        margin-right: auto;
    }

    .user-msg {
        display: flex; justify-content: flex-end;
        gap: 10px; align-items: flex-end;
        padding: 6px 0;
    }
    .user-bubble {
        background: var(--accent);
        color: #FAFAFA;
        border-radius: 16px 16px 4px 16px;
        padding: 11px 16px; font-size: 14px;
        line-height: 1.6; max-width: 72%;
        word-wrap: break-word;
        box-shadow: var(--shadow-sm);
    }
    .assistant-msg {
        display: flex; gap: 10px;
        align-items: flex-end; padding: 6px 0;
    }
    .assistant-bubble {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 4px 16px 16px 16px;
        padding: 11px 16px; font-size: 14px;
        line-height: 1.7; color: var(--text-primary);
        max-width: 80%; word-wrap: break-word;
        box-shadow: var(--shadow-sm);
    }
    .avatar {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center;
        justify-content: center; font-size: 11px;
        font-weight: 600;
        flex-shrink: 0;
    }
    .bot-avatar { background: var(--accent-soft); border: 1px solid rgba(99,102,241,0.35); color: var(--accent); }
    .user-avatar { background: var(--bg-elevated); border: 1px solid var(--border-strong); color: var(--text-secondary); }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 110px 20px 40px;
    }
    .empty-icon {
        width: 52px; height: 52px;
        background: var(--accent-soft);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: var(--radius-md);
        display: inline-flex; align-items: center;
        justify-content: center; font-size: 15px;
        font-weight: 700; color: var(--accent);
        margin-bottom: 18px;
    }
    .empty-title {
        font-size: 17px; font-weight: 600;
        color: var(--text-primary); margin-bottom: 6px;
    }
    .empty-sub { font-size: 13.5px; color: var(--text-tertiary); }

    /* ── Fixed input bar ── */
    .fixed-input-bar {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: rgba(10, 10, 11, 0.9);
        backdrop-filter: blur(12px);
        border-top: 1px solid var(--border-subtle);
        padding: 12px 24px 16px;
        z-index: 999;
    }
    .input-inner {
        max-width: 760px;
        margin: 0 auto;
        display: flex; gap: 8px; align-items: center;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
    }

    /* Text inputs / selects */
    .stTextInput input, .stTextArea textarea {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
        padding: 9px 12px !important;
        font-size: 13.5px !important;
        transition: border-color .15s, box-shadow .15s !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
        outline: none !important;
    }
    .stTextInput input::placeholder { color: var(--text-tertiary) !important; }

    div[data-baseweb="select"] > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-size: 13px !important;
    }
    div[data-baseweb="select"] * { color: var(--text-primary) !important; }
    div[data-baseweb="popover"] { background: var(--bg-elevated) !important; }
    ul[role="listbox"] { background: var(--bg-elevated) !important; border: 1px solid var(--border-subtle) !important; }
    li[role="option"] { color: var(--text-primary) !important; }
    li[role="option"]:hover { background: var(--bg-tertiary) !important; }

    label, .stMarkdown p, .stCaption {
        color: var(--text-secondary) !important;
    }

    /* Buttons */
    .stButton button {
        background: var(--accent) !important;
        color: #FAFAFA !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 9px 18px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background .15s ease, transform .1s ease !important;
        white-space: nowrap !important;
        box-shadow: var(--shadow-sm);
    }
    .stButton button:hover {
        background: var(--accent-hover) !important;
    }
    .stButton button:active {
        transform: scale(0.98);
    }

    /* Sidebar buttons — secondary style */
    [data-testid="stSidebar"] .stButton button {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: var(--bg-elevated) !important;
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    .stDownloadButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* Toggle */
    [data-testid="stToggle"] label { color: var(--text-secondary) !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--accent) !important; }
    .stSpinner p { color: var(--text-tertiary) !important; }

    /* Alerts: success / warning / error / info */
    div[data-testid="stAlertContainer"] {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-subtle) !important;
        font-size: 13px !important;
    }
    div[data-baseweb="notification"] { border-radius: var(--radius-sm) !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 10px 12px !important;
        transition: border-color .15s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-strong);
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 17px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: 10.5px !important;
        color: var(--text-tertiary) !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* Comparison result panel */
    .compare-panel {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 20px 22px;
        margin-top: 12px;
        box-shadow: var(--shadow-sm);
    }
    .compare-panel-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-bottom: 12px;
    }
    .compare-doc-chip {
        display: inline-block;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 5px 10px;
        font-size: 12.5px;
        color: var(--text-secondary);
    }
    .compare-topic {
        font-size: 12.5px;
        color: var(--text-tertiary);
        margin: 10px 0 12px;
    }
    .compare-topic strong { color: var(--text-secondary); }
    .compare-body {
        font-size: 14px;
        line-height: 1.7;
        color: var(--text-primary);
    }

    /* Agent workflow step */
    .agent-step-title {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }

    /* Info banner (missing key) */
    .status-banner {
        display: flex; align-items: center; gap: 8px;
        background: var(--danger-soft);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        font-size: 12px;
        color: #FCA5A5;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── AUTH ─────────────────────────────────────────────────────────────────────
# with open("auth_config.yaml") as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days']
# )

# try:
#     authenticator.login()
# except Exception as e:
#     st.error(f"Login error: {e}")
#     st.stop()

# if st.session_state.get("authentication_status") is False:
#     st.error("Wrong username or password.")
#     st.stop()
# elif st.session_state.get("authentication_status") is None:
#     st.warning("Please log in to continue.")
#     st.stop()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = None
if "agent_steps" not in st.session_state:
    st.session_state.agent_steps = None
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


def add_to_memory(user_msg, assistant_msg):
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    st.session_state.chat_history = st.session_state.chat_history[-20:]


def get_history_as_text():
    return "\n".join([f"{m['role'].upper()}: {m['content']}"
                      for m in st.session_state.chat_history])


# ── INIT ─────────────────────────────────────────────────────────────────────
vector_store = ChromaVectorStore()
retriever = Retriever(vector_store)
evaluator = Evaluator()

try:
    llm = GeminiLLM()
except Exception as e:
    llm = None

agent = ResearchAgent(retriever, llm) if llm else None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-block">
        <div class="brand-mark">AI</div>
        <div>
            <div class="brand-name">Research Assistant</div>
            <div class="brand-sub">Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Welcome
    name = st.session_state.get("name", "User")
    initial = name[0].upper() if name else "U"
    st.markdown(f'''
    <div class="user-strip">
        <div class="user-strip-avatar">{initial}</div>
        <div>
            <div class="user-strip-name">{name}</div>
            <div class="user-strip-role">Signed in</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    # authenticator.logout("Logout", "sidebar")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.last_scores = None
            st.session_state.agent_steps = None
            st.session_state.comparison_result = None
            st.rerun()

    # 8. Export Chat
    with col_export:
        export_clicked = st.button("Export PDF")

    if export_clicked:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()

        user_style = ParagraphStyle('user', parent=styles['Normal'],
                                     textColor=colors.HexColor('#1D4ED8'),
                                     spaceAfter=4, fontName='Helvetica-Bold')
        bot_style = ParagraphStyle('bot', parent=styles['Normal'],
                                    textColor=colors.HexColor('#111827'),
                                    spaceAfter=4)

        story = [Paragraph("AI Research Assistant — Chat Export", styles['Title']),
                 Spacer(1, 10*mm)]

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                story.append(Paragraph(f"You: {msg['content']}", user_style))
            else:
                story.append(Paragraph(f"Assistant: {msg['content']}", bot_style))
            story.append(Spacer(1, 3*mm))

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="Download PDF",
            data=buffer,
            file_name="chat_export.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # 2. Upload
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files_list:
                with st.spinner(f"Processing {uploaded_file.name}…"):
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
                        retriever.add_to_hybrid(documents)
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                        st.success(f"Indexed: {uploaded_file.name}")
                        trace_document_upload(
                            user=st.session_state.get("name", "unknown"),
                            filename=uploaded_file.name,
                            chunks=len(documents)
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")

    # # 3. Indexed Docs
    # if st.session_state.uploaded_files_list:
    #     st.markdown("### Indexed Documents")
    #     for fname in st.session_state.uploaded_files_list:
    #         st.markdown(f'<div class="file-item">{fname}</div>', unsafe_allow_html=True)
    #     st.markdown("---")

    # 5. Evaluation
    if st.session_state.last_scores:
        st.markdown("### Evaluation")
        scores = st.session_state.last_scores
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Relevance", f"{scores['relevance']}%")
            st.metric("ROUGE-1", f"{scores['rouge1']}%")
        with col_b:
            st.metric("ROUGE-L", f"{scores['rougeL']}%")
            st.metric("Coverage", f"{scores['context_coverage']}%")
        st.markdown("---")

    # 4. Compare Docs
    st.markdown("### Compare Documents")
    if len(st.session_state.uploaded_files_list) >= 2:
        doc1 = st.selectbox("Document 1", st.session_state.uploaded_files_list, key="doc1")
        doc2 = st.selectbox("Document 2", st.session_state.uploaded_files_list, key="doc2")
        compare_query = st.text_input("Topic:", key="compare_query")
        if st.button("Compare"):
            if doc1 == doc2:
                st.warning("Select different documents")
            elif compare_query:
                with st.spinner("Comparing..."):
                    results1 = [r for r in retriever.retrieve(compare_query, top_k=3)
                                if r["metadata"].get("source") == doc1]
                    results2 = [r for r in retriever.retrieve(compare_query, top_k=3)
                                if r["metadata"].get("source") == doc2]
                    context1 = "\n".join([r["text"] for r in results1]) or "No content found"
                    context2 = "\n".join([r["text"] for r in results2]) or "No content found"
                    prompt = (
                        f"Compare these two documents on: '{compare_query}'\n\n"
                        f"Doc 1 ({doc1}):\n{context1}\n\n"
                        f"Doc 2 ({doc2}):\n{context2}\n\n"
                        "Give structured comparison with similarities and differences."
                    )
                    response = llm.generate_answer(compare_query, results1 + results2, prompt)
                    st.session_state.comparison_result = response
                    st.session_state.comparison_docs = (doc1, doc2)
                    st.session_state.comparison_topic = compare_query
    else:
        st.info("Upload 2+ documents to compare")

    st.markdown("---")

    # 6. Agentic Mode
    st.markdown("### Agentic Mode")
    agentic_mode = st.toggle("Enable Agentic Workflow", value=False)

    st.markdown("---")

    # Agent steps
    if st.session_state.agent_steps:
        with st.expander("Agent Workflow Steps", expanded=True):
            for s in st.session_state.agent_steps:
                st.markdown(f'<div class="agent-step-title">{s["step"]}</div>', unsafe_allow_html=True)
                st.caption(s['result'])
                st.markdown("---")

    # error show in end
    if llm is None:
        st.markdown('<div class="status-banner">API key missing</div>', unsafe_allow_html=True)

# ── FIXED HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="fixed-header">
    <div class="fixed-header-inner">
        <div class="fixed-header-title">AI Research Assistant</div>
        <div class="fixed-header-sub">Ask questions about your uploaded documents</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# Empty state
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">AI</div>
        <div class="empty-title">How can I help you?</div>
        <div class="empty-sub">Upload a document from the sidebar and ask your first question.</div>
    </div>
    """, unsafe_allow_html=True)

# Messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="user-bubble">{msg["content"]}</div>
            <div class="avatar user-avatar">U</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-msg">
            <div class="avatar bot-avatar">AI</div>
            <div class="assistant-bubble">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

# Comparison result
if st.session_state.comparison_result:
    d1, d2 = st.session_state.comparison_docs
    st.markdown('<div class="compare-panel">', unsafe_allow_html=True)
    st.markdown('<div class="compare-panel-title">Comparison Result</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<span class="compare-doc-chip">{d1}</span>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<span class="compare-doc-chip">{d2}</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="compare-topic">Topic: <strong>{st.session_state.comparison_topic}</strong></div>',
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="compare-body">{st.session_state.comparison_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Clear Comparison"):
        st.session_state.comparison_result = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── FIXED INPUT BAR ───────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your documents…")

if question:
    submit = True
else:
    submit = False

# ── SUBMIT LOGIC ──────────────────────────────────────────────────────────────
if question:
    if llm is None:
        st.error("API key missing.")
    elif not st.session_state.uploaded_files_list:
        st.warning("Please upload at least one document first.")
    else:
        with st.spinner("Thinking..."):
            if agentic_mode:
                if agent is None:
                    st.error("Agent unavailable.")
                else:
                    steps, answer = agent.run(question)
                    scores = evaluator.evaluate(question, answer, retriever.retrieve(question, top_k=3))
                    st.session_state.last_scores = scores
                    st.session_state.agent_steps = steps
                    add_to_memory(question, answer)
                    trace_query(
                        user=st.session_state.get("name", "unknown"),
                        question=question,
                        answer=answer,
                        scores=scores,
                        agentic_mode=True
                    )
                    st.rerun()
            else:
                st.session_state.agent_steps = None
                results = retriever.retrieve(question, top_k=3)
                if not results:
                    st.warning("No relevant content found.")
                else:
                    answer = llm.generate_answer(
                        question, results, get_history_as_text()
                    )
                    scores = evaluator.evaluate(question, answer, results)
                    st.session_state.last_scores = scores
                    add_to_memory(question, answer)
                    trace_query(
                        user=st.session_state.get("name", "unknown"),
                        question=question,
                        answer=answer,
                        scores=scores,
                        agentic_mode=False
                    )
                    st.rerun()
