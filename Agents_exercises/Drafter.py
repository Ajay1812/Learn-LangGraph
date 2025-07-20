from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
import json
load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document with provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n {document_content}"

@tool
def save(filename:str) -> str:
    """Save current document to a text file and finish the process
    
    Args:
        filename: Name of the text file.
    """
    
    global document_content
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write(document_content)
        print(f"\n Document has been saved to this file {filename}")
        return f"\n Document has been saved to this file {filename}"
    except Exception as e:
        return f"Error saveing document: {str(e)}"

tools = [update, save]
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite-preview-06-17",
    api_key = os.environ["GEMINI_API_KEY"]
).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.

        The current document content is:{document_content}
        """)
    if not state["messages"]:
        user_input = "I here to help you to update a document. what would you like to create?"
        user_message = HumanMessage(content=user_input)
        
    else:
        user_input = input("What would you like to do with document? ")
        print(f"\n USERL: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n USING TOOLS: {[tc["name"] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> AgentState:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    if not messages:
        return "continue"
    
    # for most recent tool message
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in more readable format"""
    if not messages:
        return ""
    for message in messages[-3:]:
        if isinstance(messages, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges("tools", should_continue, {
    "continue" : "agent",
    "end" : END
})

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====") 
    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =====")
    
if __name__ == "__main__":
    run_document_agent()