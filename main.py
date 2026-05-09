from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from google.api_core.exceptions import ResourceExhausted


# ─────────────────────────────────────────────
# 1. Load config
# ─────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PERSIST_DIR     = config["persist_directory"]
COLLECTION_NAME = config["collection_name"]
EMBEDDING_MODEL = config["embedding_model"]
GOOGLE_API_KEY  = config.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ─────────────────────────────────────────────
# 2. RAG tool
# ─────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

@tool
def rag_tool(query: str) -> str:
    """
    Search the Transfer of Property Act 1882 knowledge base and return
    the most relevant excerpts for the given query.
    Call this whenever the user asks about property law, the Transfer of
    Property Act, legal definitions, sections, or related legal concepts.
    """
    docs = vector_store.similarity_search(query, k=4)
    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        results.append(f"[Excerpt {i} | Source: {source} | Page: {page}]\n{doc.page_content}")

    return "\n\n".join(results)


# ─────────────────────────────────────────────
# 3. LLM + tool binding
# ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a knowledgeable legal assistant specializing in the
Transfer of Property Act, 1882 (India). You have access to a RAG tool that
searches the full text of the Act. Always call the RAG tool when the user asks
about specific sections, definitions, legal concepts, or anything related to
property law under this Act. Provide clear, accurate, and well-structured
answers based on the retrieved content."""


# ─────────────────────────────────────────────
# 4. LangGraph nodes
# ─────────────────────────────────────────────
def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────
# 5. Build the graph
# ─────────────────────────────────────────────
tool_node = ToolNode(tools)

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")


# ─────────────────────────────────────────────
# 6. SQLite persistent memory
# ─────────────────────────────────────────────
DB_PATH = "chat_memory.db"
conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
memory  = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# 7. FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Transfer of Property Act — Legal Assistant",
    description="RAG-powered chatbot for the Transfer of Property Act 1882",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    username: str
    password: str
    query: str

class QueryResponse(BaseModel):
    thread_id: str
    answer: str


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    # Basic validation
    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty.")
    if not request.password.strip():
        raise HTTPException(status_code=400, detail="Password cannot be empty.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    thread_id = f"{request.username.strip()}_{request.password.strip()}"
    config_   = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config_,
        )
        answer = result["messages"][-1].content

    except ResourceExhausted:
        raise HTTPException(
            status_code=429,
            detail="Can't call LLM — Gemini API rate limit reached. Please try again later.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    return QueryResponse(thread_id=thread_id, answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}