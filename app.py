import streamlit as st
import os

# SECTION 1 — Page config (must be first Streamlit call)
st.set_page_config(
  page_title="SBI MF Facts Assistant",
  page_icon="📊",
  layout="centered",
  initial_sidebar_state="collapsed"
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
    # Load vector store from vector_store/mf_faq.json, initialize Qdrant, load cross-encoder
    _get_qdrant()
    _get_reranker()
    return True

if "backend_initialized" not in st.session_state:
    with st.spinner("Setting up retrieval engine..."):
        initialize_backend()
    st.session_state.backend_initialized = True
    st.success("Ready")

# SECTION 3 — Header
st.title("SBI MF Facts Assistant")
st.subheader("Factual answers on SBI Large Cap, Flexi Cap and ELSS Tax Saver schemes.")
st.info("Facts only. No investment advice. Sources: SBIMF / AMFI / SEBI. Do not share PAN, Aadhaar or account numbers.", icon="ℹ️")

# SECTION 5 — Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    if st.button("Expense ratio of SBI Large Cap?"):
        st.session_state.preset_query = "Expense ratio of SBI Large Cap?"
with cols[1]:
    if st.button("ELSS lock-in period?"):
        st.session_state.preset_query = "ELSS lock-in period?"
with cols[2]:
    if st.button("How to download capital gains statement?"):
        st.session_state.preset_query = "How to download capital gains statement?"

# Use the preset query if a button was clicked
user_input = st.chat_input("Ask a factual question about SBI MF schemes")
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
                    chunks = retrieve(user_input, api_key=OPENAI_API_KEY)
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
st.caption("Facts only. No investment advice. Data sourced from official public documents: sbimf.com, amfiindia.com, sebi.gov.in. Mutual fund investments are subject to market risks. Read all scheme documents carefully.")
