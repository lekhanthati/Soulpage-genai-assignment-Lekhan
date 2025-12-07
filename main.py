import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
import dotenv


dotenv.load_dotenv()


search_tool = TavilySearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tools = [search_tool, wikipedia, arxiv]

# Using an old model so it can use the tools for fetching any latest data 
agent = create_agent(
    "gpt-4-turbo-2024-04-09",
    tools=tools,
    checkpointer=InMemorySaver(),
)


st.title("🔍 LangChain Agent ")

st.write("A simple frontend to chat with your LangChain Agent using Search, Wikipedia & ArXiv tools.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])


prompt = st.chat_input("Type your message…")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)


    result = agent.invoke(
        {"messages": st.session_state.messages},
        {"configurable": {"thread_id": "1"}},
    )

    assistant_reply = result["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.chat_message("assistant").markdown(assistant_reply)
