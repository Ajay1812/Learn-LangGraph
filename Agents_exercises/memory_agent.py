from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Union
import os
load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite-preview-06-17",
    api_key=os.environ["GEMINI_API_KEY"]
)

def process_node(state: AgentState) -> AgentState:
    """This node will help to solve your query"""
    prompt = f"You are an helpful assistant. here's the user query: \n {state['messages']}"
    response = model.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    print(response.content)
    # print("Current_state: ", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process_node", process_node)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
workflow = graph.compile()

consversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    consversation_history.append(HumanMessage(content=user_input))
    result = workflow.invoke({ "messages": consversation_history})
    consversation_history = result['messages']
    user_input = input("Enter: ")

with open("history.txt", "w") as f:
    f.write("You conversation Log: \n")
    for message in consversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: , {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: , {message.content}\n\n")
    f.write("End of the conversation")