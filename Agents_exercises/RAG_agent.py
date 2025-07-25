from langgraph.graph import StateGraph, START, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    api_key = os.environ["GEMINI_API_KEY"],
    temperature = 0
)

embeddinng_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

pdf_path = "Agents_exercises/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"pdf not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Pdf is loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error pdf loading: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = "/Users/ajay/Documents/Personal/Courses/Learn-LangGraph/Agents_exercises"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddinng_model,
    persist_directory=persist_directory)
    
    vector_store.add_documents(pages_split)
    print(f"Vector stored created and collection name : {collection_name}")
except Exception as e:
    print(f"Error occured during creation of vector store: {str(e)}")
    raise

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool search information and return the relevant information from the stock market performace 2024 document.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "i found no relevant information in given document"
    result = []
    for i,doc in enumerate(docs):
        result.append(f"Document {i+1}: \n {doc.page_content} ")
    
    return "\n\n".join(result)

tools = [retriever_tool]
model = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state: AgentState) -> AgentState:
    """Check if the last message contain tool call"""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Call the llm with current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = model.invoke(messages)
    return {"messages" : state["messages"] + [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if not t['name'] in tools_dict: 
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] 
        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()