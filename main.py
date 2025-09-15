from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

