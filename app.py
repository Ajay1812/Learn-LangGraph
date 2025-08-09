from backend import chatbot
from langchain_core.messages import HumanMessage
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", cl.context.session.id)

@cl.on_message
async def on_message(msg: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    # cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in chatbot.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(**config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "chatnode"
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()