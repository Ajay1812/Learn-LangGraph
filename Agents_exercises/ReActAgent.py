from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """Simple Addition function that add two numbers"""
    return a + b

@tool
def search_browser(query):
    """This function will search browser"""
    search = DuckDuckGoSearchRun()
    response = search.invoke(query)
    return response

@tool
def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b

tools = [add, search_browser, multiply]
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite-preview-06-17",
    api_key = os.environ["GEMINI_API_KEY"]
).bind_tools(tools)

def model_call(state: AgentState) ->  AgentState:
    system_prompt = SystemMessage(content= "You are helpful AI assistant, please answer my query as best you can.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def check_condition(state: AgentState):
    messages = state["messages"]
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        return "end"
    else: 
        return "continue" 

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", check_condition, {
    "continue" : "tools",
    "end": END
})

graph.add_edge("tools", "our_agent")
app = graph.compile()

# result = app.invoke({"messages" : [("user", "Add 50 + 40 and then multiply the result by 3 and also let me know who's nikola tesla in short")]})
# print(result)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input_state = {
    "messages": [
        HumanMessage(content="Add 50 + 40 and then multiply the result by 3 and also let me know who Nikola Tesla is in short.")
    ]
}

print("\nStream Output:")
print_stream(app.stream(input=input_state, stream_mode="values"))