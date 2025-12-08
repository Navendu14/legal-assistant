import streamlit as st
import yaml
import os
from db import init_db, save_message, get_chat_history
from auth_db import init_user_db, create_user, authenticate_user, generate_chat_id

# -------------------- INITIALIZE DATABASES --------------------

init_db()
init_user_db()

# -------------------- CONFIG LOADING --------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]

# -------------------- AUTH STATE --------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = None

# ✅ NEW: Track which user's chat is currently loaded in UI
if "last_loaded_chat_id" not in st.session_state:
    st.session_state.last_loaded_chat_id = None

# -------------------- LOGIN / SIGNUP PAGE --------------------

if not st.session_state.authenticated:
    st.set_page_config(page_title="🔐 Legal AI Login")

    st.title("🔐 Legal AI Assistant Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.chat_id = generate_chat_id(username, password)

                # ✅ RESET AGENT MEMORY ONLY (NOT UI)
                from langchain_classic.memory import ConversationBufferWindowMemory
                st.session_state.memory = ConversationBufferWindowMemory(
                    k=2,
                    memory_key="chat_history",
                    return_messages=True
                )

                # ✅ FORCE CHAT RELOAD AFTER LOGIN
                st.session_state.last_loaded_chat_id = None
                st.session_state.last_processed_input = None

                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if create_user(new_user, new_pass):
                st.success("✅ Account created! You can now login.")
            else:
                st.error("❌ Username already exists")

    st.stop()

# -------------------- VECTOR DB + EMBEDDINGS --------------------

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = Chroma(
    collection_name="Legal_document",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# -------------------- LLM --------------------

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

# -------------------- TOOL --------------------

from langchain.tools import tool

@tool
def legal_retriever_tool(query: str) -> str:
    """
    Search and retrieve relevant legal documents from the vector database.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant legal documents found."

    combined_text = "\n\n".join(doc.page_content for doc in docs)
    return combined_text

tools = [legal_retriever_tool]

# -------------------- MEMORY (SESSION ONLY) --------------------

from langchain_classic.memory import ConversationBufferWindowMemory

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True
    )

memory = st.session_state.memory

# -------------------- PROMPT --------------------

from langchain_classic.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=[
        "input",
        "agent_scratchpad",
        "chat_history",
        "tools",
        "tool_names"
    ],
    template="""
You are a highly intelligent and precise legal AI assistant.

You may use tools ONLY if document lookup is required.
If the question is general legal knowledge, answer directly.

Tools:
{tools}

Format:

Question: the input question
Thought: reasoning
Action: only if needed, one of [{tool_names}]
Action Input: input to the action
Observation: tool result
Thought: I now know the final answer
Final Answer: the final answer

Previous conversation:
{chat_history}

Begin.

Question: {input}
Thought: {agent_scratchpad}
"""
)

# -------------------- AGENT --------------------

from langchain_classic.agents import create_react_agent, AgentExecutor

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="⚖️ Legal AI Chatbot")
st.title("⚖️ Legal AI Assistant")

# -------------------- LOAD CHAT HISTORY (UI ONLY, PER USER) ✅ FIXED --------------------

if st.session_state.chat_id != st.session_state.last_loaded_chat_id:
    st.session_state.messages = []

    old_messages = get_chat_history(st.session_state.chat_id)

    for role, msg in old_messages:
        st.session_state.messages.append({
            "role": role,
            "content": msg
        })

    # ✅ Mark this user as loaded
    st.session_state.last_loaded_chat_id = st.session_state.chat_id

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.subheader("Account")

    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.session_state.last_processed_input = None
        st.session_state.last_loaded_chat_id = None

        # ✅ Reset agent memory
        st.session_state.memory = ConversationBufferWindowMemory(
            k=2,
            memory_key="chat_history",
            return_messages=True
        )

        st.rerun()

# -------------------- DISPLAY CHAT --------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- INPUT --------------------

user_input = st.chat_input("Type your legal question here...")

# ✅ HARD EXECUTION LOCK (PREVENTS OLD CHAT FROM RE-TRIGGERING AGENT)
if user_input and user_input != st.session_state.last_processed_input:
    st.session_state.last_processed_input = user_input

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(st.session_state.chat_id, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                final_answer = response["output"]

                st.markdown(final_answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer
                })

                save_message(st.session_state.chat_id, "assistant", final_answer)

            except Exception as e:
                st.error(str(e))
