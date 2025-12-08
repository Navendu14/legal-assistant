import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]

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

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

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

from langchain_classic.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=2,                      # ✅ last 2 messages only
    memory_key="chat_history",
    return_messages=True
)


from langchain_classic.agents import create_react_agent

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
You are a highly intelligent and precise legal AI assistant. You can use tools to retrieve legal documents when needed.

You have access to the following tools:
{tools}

You must follow this exact format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You also have access to the previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
)



tools = [legal_retriever_tool]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

from langchain_classic.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,   # shows reasoning steps
    handle_parsing_errors=True,
)



# query = "What are the essential elements of a valid contract?"
#
# response = agent_executor.invoke({
#     "input": query
# })
#
# print("\n✅ Final Answer:\n")
# print(response["output"])

print("\n🤖 Legal Chatbot Ready with Memory! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Chatbot: Goodbye!")
        break

    try:
        response = agent_executor.invoke({
            "input": user_input
        })

        print("\n🤖 Chatbot:\n")
        print(response["output"])
        print("-" * 80)

    except Exception as e:
        print("⚠️ Error:", str(e))



