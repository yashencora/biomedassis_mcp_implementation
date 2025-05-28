import asyncio
import streamlit as st
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment and OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("medical_assistant.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# LLM for router
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# UI Setup: NextChat Style
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Inject NextChat-style CSS
st.markdown("""
<style>
body {
    background-color: #0f1117;
    color: white;
}
.chat-box {
    background-color: #1e1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.user-message {
    background-color: #2a2b32;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 5px;
    color: #fff;
}
.bot-message {
    background-color: #3a3b42;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #fff;
}
.sidebar-title {
    color: #58a6ff;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>Medical Assistant</div>", unsafe_allow_html=True)
st.sidebar.markdown("Ask a medical question related to **headaches**, **conjunctivitis**, or **COVID-19**.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Model**: `gpt-4o`")
st.sidebar.markdown("**Mode**: Medical")

# Title and input
st.title("🩺 Medical Assistant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your medical question here...")

ROUTER_PROMPT = """
You are a routing agent. Based on the medical query below, decide which domain-specific agent should handle it.
Agents:
- covid_agent: Handles questions about COVID-19, symptoms, treatments, etc.
- headaches_agent: Handles questions about migraines, tension headaches, cluster headaches, head pain, visual aura, etc.
- conjunctivitis_agent: Handles questions about pink eye, conjunctivitis, eye discharge, redness, or allergies.
If the query does not match any of these, return 'fallback_agent'.

Query: "{query}"

Answer with only the agent name: covid_agent, headaches_agent, conjunctivitis_agent, or fallback_agent.
"""

def get_bot_response(query):
    async def process():
        server_params = StdioServerParameters(command="python", args=["server.py"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Routing
                routing_prompt = ROUTER_PROMPT.format(query=query)
                routing_response = llm.invoke(routing_prompt).content.strip().lower()
                logger.info(f"[Router] Chose agent: {routing_response}")

                if routing_response == "covid_agent":
                    tool_name = "get_covid_info"
                elif routing_response == "headaches_agent":
                    tool_name = "get_headache_info"
                elif routing_response == "conjunctivitis_agent":
                    tool_name = "get_conjunctivitis_info"
                else:
                    return "🤖 Sorry, I don't have information on that topic."

                # Tool call
                try:
                    response = await session.call_tool(tool_name, {"query": query})
                    if isinstance(response, tuple):
                        response = response[0]
                    if hasattr(response, 'content') and isinstance(response.content, list):
                        return "\n".join([c.text for c in response.content if c.type == 'text'])
                    return "🤖 Unable to parse response."
                except Exception as e:
                    logger.error(f"[Error] Tool invocation failed: {e}", exc_info=True)
                    return f"❌ Error calling tool: {e}"
    return asyncio.run(process())

# Handle input
if user_input:
    with st.spinner("🤖 Thinking..."):
        bot_response = get_bot_response(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_response))

# Display chat history
for role, msg in reversed(st.session_state.chat_history):
    if role == "user":
        st.markdown(f"<div class='user-message'>🧑‍💬 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>🤖 {msg}</div>", unsafe_allow_html=True)