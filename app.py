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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    * { box-sizing: border-box; margin: 0; padding: 0; }

    .stApp { background-color: #F7F7F8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5E5;
    }
    [data-testid="stSidebar"] * { color: #111827 !important; }
    [data-testid="stSidebar"] h3 {
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #9CA3AF !important;
        margin: 0 0 8px !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] > div {
        border: 1.5px dashed #D1D5DB !important;
        border-radius: 10px !important;
        background: #F9FAFB !important;
        padding: 12px !important;
    }

    /* File item */
    .file-item {
        display: flex; align-items: center; gap: 6px;
        background: #F0FDF4; border: 1px solid #BBF7D0;
        border-radius: 8px; padding: 5px 10px;
        margin: 3px 0; font-size: 12px; color: #15803D;
    }

    /* Welcome strip */
    .welcome-strip {
        display: flex; align-items: center; gap: 8px;
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 10px; padding: 8px 12px;
        font-size: 13px; color: #1D4ED8; margin-bottom: 4px;
    }

    /* Fixed header */
    .fixed-header {
        position: fixed;
        top: 0; left: 0; right: 0;
        background: rgba(247, 247, 248, 0.95);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #E5E5E5;
        z-index: 999;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .fixed-header-title {
        font-size: 16px;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.01em;
    }
    .fixed-header-sub {
        font-size: 12px;
        color: #9CA3AF;
        margin-top: 2px;
        text-align: center;
    }

    /* Chat area */
    .chat-wrapper {
        margin-top: 80px;
        margin-bottom: 100px;
        padding: 0 16px;
        max-width: 760px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Messages */
    .user-msg {
        display: flex; justify-content: flex-end;
        gap: 10px; align-items: flex-end;
        padding: 6px 0;
    }
    .user-bubble {
        background: #111827; color: #F9FAFB;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px; font-size: 14px;
        line-height: 1.6; max-width: 72%;
        word-wrap: break-word;
    }
    .assistant-msg {
        display: flex; gap: 10px;
        align-items: flex-end; padding: 6px 0;
    }
    .assistant-bubble {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 4px 18px 18px 18px;
        padding: 10px 16px; font-size: 14px;
        line-height: 1.7; color: #111827;
        max-width: 80%; word-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .avatar {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center;
        justify-content: center; font-size: 13px;
        flex-shrink: 0;
    }
    .bot-avatar { background: #EFF6FF; border: 1px solid #BFDBFE; }
    .user-avatar { background: #111827; }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 100px 20px 40px;
    }
    .empty-icon {
        width: 56px; height: 56px;
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 16px;
        display: inline-flex; align-items: center;
        justify-content: center; font-size: 26px;
        margin-bottom: 16px;
    }
    .empty-title {
        font-size: 18px; font-weight: 600;
        color: #111827; margin-bottom: 8px;
    }
    .empty-sub { font-size: 14px; color: #6B7280; }

    /* Fixed input bar */
    .fixed-input-bar {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: rgba(247, 247, 248, 0.97);
        backdrop-filter: blur(10px);
        border-top: 1px solid #E5E5E5;
        padding: 12px 24px 16px;
        z-index: 999;
    }
    .input-inner {
        max-width: 760px;
        margin: 0 auto;
        display: flex; gap: 8px; align-items: center;
    }

    /* Input styling */
    .stTextInput input {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        transition: border-color .15s, box-shadow .15s !important;
    }
    .stTextInput input:focus {
        border-color: #6B7280 !important;
        box-shadow: 0 0 0 3px rgba(107,114,128,.12) !important;
        outline: none !important;
    }

    /* Send button */
    .stButton button {
        background: #111827 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background .15s !important;
        white-space: nowrap !important;
    }
    .stButton button:hover {
        background: #374151 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        color: #6B7280 !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #F9FAFB !important;
    }

    hr { border-color: #E5E5E5 !important; margin: 10px 0 !important; }
    .stSpinner > div { border-top-color: #111827 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #F9FAFB;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 8px !important;
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
    # 1. Welcome
    name = st.session_state.get("name", "User")
    st.markdown(f'<div class="welcome-strip">👤 <strong>{name}</strong></div>',
                unsafe_allow_html=True)
    # authenticator.logout("🚪 Logout", "sidebar")

    
    # Clear
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_scores = None
        st.session_state.agent_steps = None
        st.session_state.comparison_result = None
        st.rerun()
          
    # 8. Export Chat

    if st.session_state.chat_history:
        st.markdown("### 📥 Export")
    
    if st.button("📄 Export Chat PDF"):
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
            label="⬇️ Download PDF",
            data=buffer,
            file_name="chat_export.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # 2. Upload
    st.markdown("### 📁 Documents")
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
                        st.success(f"✅ {uploaded_file.name}")
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
    #     st.markdown("### 📚 Indexed Documents")
    #     for fname in st.session_state.uploaded_files_list:
    #         st.markdown(f'<div class="file-item">📄 {fname}</div>', unsafe_allow_html=True)
    #     st.markdown("---")

    # 5. Evaluation
    if st.session_state.last_scores:
        st.markdown("### 📊 Evaluation")
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
    st.markdown("### 🔍 Compare Documents")
    if len(st.session_state.uploaded_files_list) >= 2:
        doc1 = st.selectbox("Document 1", st.session_state.uploaded_files_list, key="doc1")
        doc2 = st.selectbox("Document 2", st.session_state.uploaded_files_list, key="doc2")
        compare_query = st.text_input("Topic:", key="compare_query")
        if st.button("🔄 Compare"):
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
    st.markdown("### 🤖 Agentic Mode")
    agentic_mode = st.toggle("Enable Agentic Workflow", value=False)

    st.markdown("---")



    # Agent steps
    if st.session_state.agent_steps:
        with st.expander("🤖 Agent Workflow Steps", expanded=True):
            for s in st.session_state.agent_steps:
                st.markdown(f"**{s['step']}**")
                st.caption(s['result'])
                st.markdown("---")



    #error show in end 

    if llm is None:
        st.error("⚠️ API key missing")

# ── FIXED HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="fixed-header">
    <div>
        <div class="fixed-header-title">🤖 AI Research Assistant</div>
        <div class="fixed-header-sub">Ask questions about your uploaded documents</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# # Agent steps
# if st.session_state.agent_steps:
#     with st.expander("🤖 Agent Workflow Steps", expanded=True):
#         for s in st.session_state.agent_steps:
#             st.markdown(f"**{s['step']}**")
#             st.caption(s['result'])
#             st.markdown("---")

# Empty state
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🤖</div>
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
            <div class="avatar user-avatar">👤</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-msg">
            <div class="avatar bot-avatar">🤖</div>
            <div class="assistant-bubble">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

# Comparison result
if st.session_state.comparison_result:
    st.markdown("---")
    st.markdown("### 🔍 Comparison Result")
    d1, d2 = st.session_state.comparison_docs
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**📄 {d1}**")
    with c2:
        st.markdown(f"**📄 {d2}**")
    st.markdown(f"**Topic:** {st.session_state.comparison_topic}")
    st.markdown(st.session_state.comparison_result)
    if st.button("❌ Clear Comparison"):
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