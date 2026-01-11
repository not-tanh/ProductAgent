import sys
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.tools import product_analysis_tool
from agents.prompts import ANALYSIS_AGENT_PROMPT
load_dotenv()


product_analysis_agent = create_agent(
    model=os.getenv('PRODUCT_ANALYSIS_MODEL'),
    tools=[product_analysis_tool],
    system_prompt=ANALYSIS_AGENT_PROMPT
)


@tool
def product_analysis(request: str):
    """
    Search for products insights in internal database using natural language
    """
    result = product_analysis_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
