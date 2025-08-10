from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import  add_messages
from dotenv import load_dotenv
import os
import sqlite3
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-06-17",
        api_key=os.environ["GEMINI_PRO_API_KEY"]
)
system_message = SystemMessage(content="""
    You are an AI chatbot designed to provide short, concise, and accurate responses. Answer queries clearly and directly, using as few words as needed without losing meaning. Avoid unnecessary explanations, filler, or repetition. Stay relevant and focused on the user's question.
""")

def chatnode(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages).content
    return {"messages": [response]}

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node("chatnode", chatnode)
graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)
chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id": "2"}}

# response = chatbot.invoke({"messages": HumanMessage(content="What is my name?")}, config=CONFIG)
# print(response)

def retrieve_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_thread)
