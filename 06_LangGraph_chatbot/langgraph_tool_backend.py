from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests
import sqlite3

load_dotenv()

# ----------------
# 1. LLM
# ----------------

llm = ChatGoogleGenerativeAI(
    # "gemini-2.5-pro"
    model = "gemini-2.5-flash-lite-preview-06-17",
    api_key = os.environ["GEMINI_PRO_API_KEY"]
)


# ----------------
# 2. Tools
# ----------------
# Tools
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """This tool will handle basic arithematic operation on two numbers
    Operations: add (+), subtract (-), multiply (*), divide (/)
    """
    try:
        if operation == "+":
            result = first_num + second_num
        elif operation == "-":
            result = first_num - second_num
        elif operation == "*":
            result = first_num * second_num
        elif operation == "/":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA', 'IBM')
    using Alpha vantage using API"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.environ["ALPHA_VINTAGE_API_KEY"]}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools=tools)

# ----------------
# 3. States
# ----------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 


# ----------------
# 4. Nodes
# ----------------

def chat_node(state: ChatState):
    """LLM node that answer your query or request a tool call"""
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages": [response]}

tool_node = ToolNode(tools)


# ----------------
# 5. Checkpointer
# ----------------

conn = sqlite3.connect(database="db/chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ----------------
# 6. Graph
# ----------------

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
# ----------------
# 7. Helper
# ----------------

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)