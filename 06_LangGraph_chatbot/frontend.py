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
        
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message =  response["messages"][-1].content
    st.session_state["messages_history"].append({'role': "assistant", "content":ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)