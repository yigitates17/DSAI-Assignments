from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
import requests
import psycopg2
import os
import json

MODEL="gemini-2.5-flash-preview-05-20"

load_dotenv()

@tool(description="Get the current weather in a given location")
def get_weather(location: str) -> str:
    return f"It's rainy in {location}."

llm = ChatGoogleGenerativeAI(model=MODEL)
llm_with_tools = llm.bind_tools([get_weather])

messages = []

query = "How is the weather in Morocco?"
user_message = HumanMessage(content=query)

messages.append(user_message)

ai_msg = llm_with_tools.invoke([user_message])

for tool_call in ai_msg.tool_calls:

    print(tool_call)

    if tool_call["name"] == "get_weather":
        result = get_weather.invoke(tool_call["args"])

    tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            
    messages.append(tool_message)

    final_response = llm_with_tools.invoke([user_message, ai_msg, tool_message])
    print("Final response:", final_response.content)