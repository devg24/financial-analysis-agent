import streamlit as st
import requests
import json
import os

# Configure the API URL. In local dev it is 127.0.0.1. 
# Via docker-compose, this is overridden to http://api:8000/chat/stream via ENV.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat/stream")

st.set_page_config(page_title="FinAgent Portfolio", page_icon="📈", layout="wide")

st.title("📈 🤖 FinAgent: Autonomous Financial AI")

with st.sidebar:
    st.markdown("### 👨‍💻 About this Agent")
    st.markdown(
        "This application uses **LangGraph** to construct a deterministic multi-agent state machine. "
        "The **Planner Agent** parses the query, while the **Supervisor** appropriately routes tasks "
        "to specialized **Quant, Fundamental, and Sentiment Agents**.\n\n"
        "Finally, the **Summarizer** compiles a comprehensive Investment Memo."
    )
    
    st.divider()
    
    # Automatically survey the indexed ChromeDB vector stores
    available_tickers = []
    if os.path.exists("./chroma_db"):
        for d in os.listdir("./chroma_db"):
            if d.endswith("_10k"):
                available_tickers.append(d.replace("_10k", ""))
                
    if available_tickers:
        st.markdown("### 📚 Supported 10-K Data")
        st.markdown("Deep RAG (Fundamental SEC filings) currently verified & compiled for:")
        
        # Display as nice Streamlit badges/tags
        cols = st.columns(4)
        for i, t in enumerate(sorted(available_tickers)):
            cols[i % 4].code(t)
            
    st.divider()
    
    st.markdown("### ⚡ Recruiter Quick-Test")
    st.markdown("Try one of these example queries to see the multi-agent graph in action:")
    
    if st.button("🍎 Apple Financial Overview"):
        st.session_state.example_query = "What is the price, sentiment, and recent 10-K risks for Apple (AAPL)?"
    
    if st.button("🏎️ Tesla Breaking Sentiment"):
        st.session_state.example_query = "What is the latest news sentiment for TSLA?"
        
    if st.button("💻 MSFT vs GOOGL"):
        st.session_state.example_query = "Compare the current stock performance of Microsoft and Google."
    
    st.divider()
    st.caption("Powered by Llama-3.1-8B via Groq")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "steps" in message and message["steps"]:
            # Render past steps in a collapsed status box
            total_time = message.get("total_latency", 0)
            title = f"✅ Investment Memo Generated! (Total Latency: {total_time}s)" if total_time else "✅ Investment Memo Generated!"
            with st.status(title, expanded=False):
                for step in message["steps"]:
                    lat = step.get('step_latency', 0)
                    lat_str = f"({lat}s) " if lat else ""
                    st.write(f"**[{step['node']}]** {lat_str}{step['content']}")
        st.markdown(message["content"])

# Unconditionally render the chat_input so it NEVER disappears from the UI
chat_val = st.chat_input("Ask about any stock ticker (e.g. AAPL, TSLA, NVDA)...")

if "example_query" in st.session_state and st.session_state.example_query:
    prompt = st.session_state.example_query
    st.session_state.example_query = "" # Reset
else:
    prompt = chat_val

if prompt:
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We will stream the intermediate node outputs into a dynamic expanding status box
        status_box = st.status("🧠 Consulting Specialized AI Agents...", expanded=True)
        final_memo_placeholder = st.empty()
        
        try:
            # Stream the response via POST request
            with requests.post(API_URL, json={"query": prompt}, stream=True) as response:
                response.raise_for_status()
                
                final_memo = ""
                session_steps = []
                
                # Consume Server-Sent Events (SSE)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[len("data: "):]
                            try:
                                data = json.loads(data_str)
                                node = data.get("node")
                                content = data.get("content")
                                step_latency = data.get("step_latency", 0)
                                total_latency = data.get("total_latency", 0)
                                
                                if node == "Summarizer":
                                    # The final node returns the full markdown report
                                    final_memo = content
                                    final_memo_placeholder.markdown(final_memo)
                                    status_box.update(label=f"✅ Investment Memo Generated! (Total Latency: {total_latency}s)", state="complete", expanded=False)
                                else:
                                    # Show what the different agents (Quant, Sentiment, etc.) are calculating
                                    lat_str = f"({step_latency}s) " if step_latency else ""
                                    status_box.write(f"**[{node}]** {lat_str}{content}")
                                    session_steps.append({
                                        "node": node, 
                                        "content": content,
                                        "step_latency": step_latency,
                                        "total_latency": total_latency
                                    })
                            except json.JSONDecodeError:
                                pass
                
                # Save the final memo and intermediate steps to history
                if final_memo:
                    final_total_latency = session_steps[-1].get("total_latency", 0) if session_steps else 0
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_memo, 
                        "steps": session_steps,
                        "total_latency": final_total_latency
                    })
                    
        except requests.exceptions.RequestException as e:
            status_box.update(label="❌ Connection Error", state="error", expanded=False)
            st.error(f"Failed to connect to the backend FastAPI server: {e}")
