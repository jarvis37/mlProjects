import streamlit as st
import time
from courtroom import run_trial
from rag import RAGRetriever

# ---------------------------------------------------------------------------
# Page Configuration & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Courtroom Simulation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, dark-mode inspired UI with beautiful typography
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #f1f5f9;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    
    /* Elegant Title Area */
    .title-area {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    
    .title-area h1 {
        margin: 0;
        font-size: 2.5rem;
        color: #ffffff;
    }
    
    .title-area p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        color: #cbd5e1;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.5;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Button Styling */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    
    /* Output Card Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0f172a;
        padding: 5px 5px 0 5px;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
        color: #94a3b8;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2e3b4e;
        color: #f8fafc;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Verdict Card */
    .verdict-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .guilty-highlight {
        color: #ef4444;
        font-size: 2rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .not-guilty-highlight {
        color: #10b981;
        font-size: 2rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #0f172a;
    }
    
    /* Status indicators */
    .status-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Management & Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_retriever():
    """Load the RAG model once and cache it in memory."""
    return RAGRetriever()

if 'trial_started' not in st.session_state:
    st.session_state.trial_started = False
if 'trial_results' not in st.session_state:
    st.session_state.trial_results = None

DEMO_CASE = (
    "A man was seen running away from a jewelry store moments after the alarm "
    "went off. Security camera footage shows a person matching his description "
    "near the scene. No stolen items were found on him. He claims he was "
    "jogging in the area and panicked when he heard the alarm."
)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/courthouse.png", width=60)
    st.title("⚖️ Settings")
    st.markdown("Configure the trial parameters below.")
    
    top_k = st.slider(
        "RAG Legal Contexts (k)", 
        min_value=1, max_value=8, value=4, 
        help="Number of relevant legal principles to retrieve and pass to the LLM agents."
    )
    
    st.divider()
    
    st.markdown("### 🤖 The Architecture")
    st.markdown("""
    This app orchestrates three independent GPT agents:
    1. **RAG Database**: FAISS retrieves `top_k` legal principles based on the case facts.
    2. **Prosecutor GPT**: Argues for guilt using context.
    3. **Defense GPT**: Defends the accused using context.
    4. **Judge GPT**: Reads both arguments and delivers a final verdict with a confidence score.
    """)
    
    st.divider()
    st.caption("Developed by [jarvis37](https://github.com/jarvis37)")


# --- MAIN AREA ---

# Header
st.markdown("""
<div class="title-area">
    <h1>AI Courtroom Simulation</h1>
    <p>Submit case details for an autonomous multi-agent trial proceeding.</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 📝 Enter Case Description")
case_input = st.text_area(
    "Provide the facts of the case, witness testimonies, or evidence found at the scene.",
    value=DEMO_CASE,
    height=150,
    placeholder="Describe the crime, the suspect, the evidence...",
    label_visibility="collapsed"
)

# Start Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_trial = st.button("⚖️ Commence Trial", use_container_width=True)

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------
if start_trial and case_input:
    st.session_state.trial_started = True
    st.session_state.trial_results = None
    
    # Create empty placeholders for streaming UI updates
    status_container = st.empty()
    
    try:
        # Load retriever (cached)
        with st.spinner("Loading RAG Database..."):
            retriever = load_retriever()
        
        # Step-by-step UI updates to show progress visually
        status_container.markdown('<div class="status-box">🔍 <b>Step 1/4:</b> Querying FAISS for legal context...</div>', unsafe_allow_html=True)
        time.sleep(1) # Small sleep for UX smoothness
        
        status_container.markdown('<div class="status-box">🔴 <b>Step 2/4:</b> Prosecutor GPT is building its argument...</div>', unsafe_allow_html=True)
        # Note: We run the whole function synchronously, so the UI won't perfectly stream
        # between agents in real-time unless we refactored courtroom.py to yield results. 
        # For now, we simulate the UI updates.
        
        # Run actual trial
        results = run_trial(case_input, retriever, top_k=top_k)
        
        status_container.markdown('<div class="status-box">🔵 <b>Step 3/4:</b> Defense GPT is cross-examining and formulating defense...</div>', unsafe_allow_html=True)
        time.sleep(1)
        
        status_container.markdown('<div class="status-box">⚖️ <b>Step 4/4:</b> Judge GPT is reviewing arguments and deliberating...</div>', unsafe_allow_html=True)
        time.sleep(1)
        
        status_container.empty()
        st.session_state.trial_results = results
        
    except Exception as e:
        status_container.empty()
        st.error(f"🚨 **Trial Error:** {str(e)}")
        st.info("💡 **Tip:** Make sure your `.env` file has a valid `OPENAI_API_KEY` set.")

# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
if st.session_state.trial_started and st.session_state.trial_results:
    res = st.session_state.trial_results
    
    st.markdown("---")
    st.markdown("## 📜 Trial Proceedings")
    
    # Use tabs for a clean, organized layout
    tab1, tab2, tab3 = st.tabs(["🔴 Prosecution", "🔵 Defense", "📚 Legal Context"])
    
    with tab1:
        st.markdown(res['prosecution'])
        
    with tab2:
        st.markdown(res['defense'])
        
    with tab3:
        st.markdown(res['legal_context'])
        
    # Highlight the verdict aggressively
    st.markdown("---")
    
    # Extract the guilty/not guilty string dynamically safely
    verdict_text = res['verdict']
    
    # Display the Judge's full statement
    st.markdown("## 👨‍⚖️ Judge's Ruling")
    st.markdown(verdict_text)

