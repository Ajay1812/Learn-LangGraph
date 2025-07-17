from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List
import os
load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite-preview-06-17",
    api_key=os.environ["GEMINI_API_KEY"]
)

def process_node(state: AgentState) -> AgentState:
    prompt = f"You are an helpful assistant. here's the user query: \n {state['messages']}"
    response = model.invoke(prompt)
    print("AI: ", response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process_node", process_node)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
workflow = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    workflow.invoke({ "messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
