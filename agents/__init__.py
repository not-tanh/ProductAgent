import os
import sys

from langchain.agents import create_agent
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agents.prompts import ORCHESTRATOR_AGENT_PROMPT, SIMPLE_AGENT_PROMPT
from agents.product_search_agent import internal_search, search_products_tool
from agents.web_analysis_agent import web_search, web_analysis_tool
from agents.tools import product_analysis_tool, get_schema_duckdb

orchestrator_agent = create_agent(
    model=os.getenv('ORCHESTRATOR_MODEL'),
    tools=[internal_search, web_search],
    system_prompt=ORCHESTRATOR_AGENT_PROMPT
)

simple_agent = create_agent(
    model=os.getenv('ORCHESTRATOR_MODEL'),
    tools=[search_products_tool, web_analysis_tool, product_analysis_tool],
    system_prompt=SIMPLE_AGENT_PROMPT.format(table_schema=get_schema_duckdb())
)
