import sys
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.tools import search_products_tool
from agents.prompts import SEARCH_AGENT_PROMPT
load_dotenv()


search_agent = create_agent(
    model=os.getenv('PRODUCT_SEARCH_MODEL'),
    tools=[search_products_tool],
    system_prompt=SEARCH_AGENT_PROMPT
)


@tool
def internal_search(request: str):
    """
    Search for products in internal database in natural language
    :param request: query in natural language
    :return: list of found products with details & summarization
    """
    result = search_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
