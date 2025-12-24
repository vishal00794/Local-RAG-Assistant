import streamlit as st
import time

# --- IMPORT YOUR AGENT ---
# We try to import the Ollama version first (since that's what you are using)
try:
    from step2_agent_ollama import BankComplianceAgent
except ImportError:
    try:
        from step2_agent import BankComplianceAgent
    except ImportError:
        st.error("❌ Critical Error: Could not find 'step2_agent_ollama.py' or 'step2_agent.py'.")
        st.stop()

# ---------------------------------------------------------
# UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SentinAL - Compliance AI",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a professional banking look
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #e8f4f9;
        border-left: 5px solid #0068c9;
    }
    h1 { color: #003057; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INITIALIZE BACKEND (Cached)
# ---------------------------------------------------------
# We use @st.cache_resource so we don't reload Llama3 every time you click a button
@st.cache_resource
def get_agent():
    print("🔄 Initializing Agent for UI...")
    return BankComplianceAgent()

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to load the AI Agent. Is Ollama running? Error: {e}")
    st.stop()

# ---------------------------------------------------------
# SESSION STATE (Chat History)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# SIDEBAR: SETTINGS & INFO
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=50)
    st.title("SentinAL 🛡️")
    st.markdown("---")
    st.markdown("**System Status:**")
    st.success("🟢 System Online")
    st.markdown(f"**Engine:** Llama 3 (Local)")
    st.markdown(f"**Database:** ChromaDB (Persisted)")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("© 2025 Internal Compliance Tool")

# ---------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("🏦 Regulatory Compliance Assistant")
st.markdown("Ask questions about *Transaction Processing Rules*, *Preauthorizations*, or *Regional Routing*.")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input
if prompt := st.chat_input("Ex: What are the routing rules for Canada?"):
    
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔍 Consulting Policy Documents..."):
            try:
                # Call the Agent
                raw_response = agent.ask(prompt)
                
                # Check for errors in response text
                if "❌" in raw_response:
                    st.error(raw_response)
                    full_response = raw_response
                else:
                    # Simulate typing effect
                    for chunk in raw_response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = f"Error: {e}"

    # Add Assistant Message to History
    st.session_state.messages.append({"role": "assistant", "content": full_response})