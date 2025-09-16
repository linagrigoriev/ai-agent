from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool

load_dotenv()

class ResearchResponse(BaseModel):      # make sure to inherit from BaseModel
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",                       # system message - tell LLM what it supposed to be doing
        """You are a research assistant that will help generate a pdf regarding a programming project workflow.
        Answer the query and use necessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions} 
        """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,                    
    tools=tools
)

agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("With what project can I help you today?\n")
raw_response = agent_executer.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_output)
except Exception as e:
    print("Error parsing response:", e)

print(structured_response.topic)