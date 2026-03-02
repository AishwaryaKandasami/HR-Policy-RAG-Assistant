import streamlit as st
import os

# SECTION 1 — Page config (must be first Streamlit call)
st.set_page_config(
  page_title="SBI MF Facts — for Groww Users",
  page_icon="📊",
  layout="centered",
  initial_sidebar_state="collapsed"
)

st.markdown(
    "<style>"
    "div.stButton > button {"
    "  border: 1px solid #00d09c;"
    "  color: #00d09c;"
    "  background-color: white;"
    "}"
    "div.stButton > button:hover {"
    "  background-color: #e6faf5;"
    "  color: #00d09c;"
    "}"
    "div.stButton > button:focus {"
    "  box-shadow: none !important;"
    "}"
    "div.stButton > button.active {"
    "  background-color: #00d09c !important;"
    "  color: white !important;"
    "  border: none !important;"
    "}"
    "a { color: #00d09c !important; }"
    "</style>",
    unsafe_allow_html=True
)

# SECTION 3 — SECRETS HANDLING FOR STREAMLIT CLOUD
# Works both locally (.env) and on Streamlit Cloud (secrets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or \
                 os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY") or \
                 os.environ.get("GROQ_API_KEY")

if not OPENAI_API_KEY or not GROQ_API_KEY:
    st.error("API keys not configured. Add OPENAI_API_KEY and GROQ_API_KEY to Streamlit secrets.")
    st.stop()

# SECTION 2 — Startup loading with spinner
from guardrails import classify_query
from retriever import _get_qdrant, _get_reranker, retrieve
from generator import generate

@st.cache_resource(show_spinner=False)
def initialize_backend():
    # Load vector store from vector_store/mf_faq.json (reloaded for Feb 2026 update)
    _get_qdrant()
    _get_reranker()
    return True

if "backend_initialized" not in st.session_state:
    with st.spinner("Setting up retrieval engine..."):
        initialize_backend()
    st.session_state.backend_initialized = True
    st.sidebar.success("✓ Knowledge base loaded")

# SECTION 3 — Header
st.title("SBI MF Facts Assistant")
st.markdown("<p style='font-size: 0.8em; color: gray; margin-top: -15px; margin-bottom: 15px;'>Built for Groww users · Powered by official AMC & SEBI data</p>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:16px; color:#444; margin-top:-10px;'>"
    "Get instant factual answers on SBI MF schemes on Groww "
    "— expense ratios, exit loads, SIP minimums, ELSS lock-in."
    "</p>",
    unsafe_allow_html=True
)
st.info("ℹ️  Facts only · No investment advice · For Groww users researching SBI MF schemes · Sources: SBIMF / AMFI / SEBI · Do not share PAN, Aadhaar or account numbers.")

# SECTION 5 — Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# SECTION 5a — Scheme Selector
if "selected_scheme" not in st.session_state:
    st.session_state.selected_scheme = "All Schemes"

st.markdown("<p style='font-size: 12px; color: gray; margin-bottom: 5px;'>Asking about:</p>", unsafe_allow_html=True)

# Layout for the 4 buttons
cols = st.columns(4)
schemes = ["All Schemes", "SBI Large Cap", "SBI Flexi Cap", "SBI ELSS"]

for i, scheme in enumerate(schemes):
    with cols[i]:
        # Determine if this button is the active one
        is_active = st.session_state.selected_scheme == scheme
        
        # Streamlit doesn't support dynamic classes on buttons natively.
        # However, we can use a trick with markdown + HTML for identical styling, 
        # or use standard st.button and rely on rerun to re-render.
        # Since the user requested CSS overriding, we will inject a dynamic style block 
        # specifically for the nth-child button if it's active.
        
        if is_active:
            st.markdown(
                f"<style>div[data-testid='column']:nth-of-type({i+1}) div.stButton > button {{ background-color: #00d09c !important; color: white !important; border: none !important; }}</style>",
                unsafe_allow_html=True
            )
            
        if st.button(scheme, key=f"btn_{scheme}", use_container_width=True):
            st.session_state.selected_scheme = scheme
            st.rerun()

if st.session_state.selected_scheme != "All Schemes":
    st.markdown(f"<p style='font-size: 12px; color: gray;'>Showing results for {st.session_state.selected_scheme} only. Select All Schemes to search across all funds.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='margin-bottom: 20px;'></p>", unsafe_allow_html=True)

# Welcome message if no chat history
if len(st.session_state.messages) == 0:
    st.markdown(
        "<div style='background:#fffbea; border-left:4px solid "
        "#f5a623; padding:12px 16px; border-radius:6px; "
        "font-size:14px; color:#555;'>"
        "👋 Welcome, Groww user. Ask me any factual question "
        "about SBI Large Cap, SBI Flexi Cap, or SBI ELSS Tax "
        "Saver Fund. Answers come with a verified source link. "
        "No investment advice or recommendations."
        "</div>",
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_label") and msg.get("source_url"):
            st.markdown(f"Source: [{msg['source_label']}]({msg['source_url']})")
        if msg.get("date_fetched"):
            st.markdown(f"Last updated from sources: {msg['date_fetched']}")

# SECTION 4 — Example question chips
cols = st.columns(3)
with cols[0]:
    if st.button("SBI Large Cap expense ratio?"):
        st.session_state.preset_query = "SBI Large Cap expense ratio?"
with cols[1]:
    if st.button("ELSS lock-in and 80C benefit?"):
        st.session_state.preset_query = "ELSS lock-in and 80C benefit?"
with cols[2]:
    if st.button("Download capital gains from CAMS?"):
        st.session_state.preset_query = "Download capital gains from CAMS?"

# Use the preset query if a button was clicked
user_input = st.chat_input("Ask about SBI MF schemes on Groww...")
preset_query = st.session_state.pop("preset_query", None)
if preset_query:
    user_input = preset_query

# SECTION 6 — Chat input and pipeline
if user_input:
    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Show st.spinner("Retrieving answer...")
    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            try:
                # 3. Call classify_query
                classification = classify_query(user_input)
                action = classification.get("action")

                if action == "retrieve":
                    if st.session_state.selected_scheme == "All Schemes":
                        chunks = retrieve(user_input, api_key=OPENAI_API_KEY, scheme_filter=None)
                    else:
                        chunks = retrieve(
                            user_input,
                            api_key=OPENAI_API_KEY,
                            scheme_filter=st.session_state.selected_scheme
                        )
                    result = generate(user_input, chunks, api_key=GROQ_API_KEY)
                    
                    answer = result["answer"]
                    source_label = result.get("source_label", "")
                    source_url = result.get("source_url", "")
                    date_fetched = result.get("date_fetched", "")

                    st.markdown(answer)
                    if source_label and source_url:
                        st.markdown(f"Source: [{source_label}]({source_url})")
                    if date_fetched:
                        st.markdown(f"Last updated from sources: {date_fetched}")
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_label": source_label,
                        "source_url": source_url,
                        "date_fetched": date_fetched
                    })

                elif action == "refuse_advice":
                    answer = "I cannot provide investment advice. Please consult a registered investment advisor."
                    link = "amfiindia.com/investor-corner"
                    st.markdown(answer)
                    st.markdown(f"[{link}](https://{link})")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer + f"\n\n[{link}](https://{link})"
                    })

                elif action == "fallback":
                    answer = "This query is outside my current scope (SBI Large Cap, Flexi Cap, ELSS only). Please visit sbimf.com for other schemes."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif action == "block_pii":
                    answer = "Please do not share personal identifiers. I can only answer factual questions about scheme features."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                # TASK 4 — ERROR HANDLING
                print(f"Pipeline error: {e}")
                st.error("Unable to retrieve answer. Please try again or visit sbimf.com directly.")

# SECTION 7 — Footer
st.markdown("---")
st.markdown(
    "<div style='font-size:11px; color:#888; "
    "padding:16px 0; line-height:1.6;'>"
    "For Groww users only &nbsp;·&nbsp; "
    "Facts sourced from sbimf.com, amfiindia.com, "
    "sebi.gov.in &nbsp;·&nbsp; "
    "This tool is not affiliated with or endorsed by Groww "
    "&nbsp;·&nbsp; "
    "Mutual fund investments are subject to market risks "
    "&nbsp;·&nbsp; "
    "Read all scheme documents carefully before investing."
    "</div>",
    unsafe_allow_html=True
)
