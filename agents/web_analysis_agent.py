import sys
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.tools import web_analysis_tool
from agents.prompts import WEB_SEARCH_AGENT_PROMPT
load_dotenv()


web_analysis_agent = create_agent(
    model=os.getenv('WEB_ANALYSIS_MODEL'),
    tools=[web_analysis_tool],
    system_prompt=WEB_SEARCH_AGENT_PROMPT
)


@tool
def web_search(request: str):
    """
    Search the internet for information related to products
    :param request: query in natural language
    :return: information found on the internet, can include summarizations, feedbacks or insights
    """
    result = web_analysis_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
