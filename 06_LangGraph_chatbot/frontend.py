import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "1"}}

if 'messages_history' not in st.session_state:
    st.session_state["messages_history"]  = [] 
    

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]): 
        st.text(message["content"])

user_input =  st.chat_input("Whay's in your mind? ")

if user_input:
    
    st.session_state["messages_history"].append({'role': "user", "content": user_input})
    with st.chat_message("human"):
        st.text(user_input)
        
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, meta_data in chatbot.stream(
                    {"messages" : HumanMessage(content=user_input)},
                    {"configurable": {"thread_id": "12"}},
                    stream_mode="messages"
            )
        )
        
    st.session_state["messages_history"].append({'role': "assistant", "content":ai_message})