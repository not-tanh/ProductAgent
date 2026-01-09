import os
import sys

from langchain.agents import create_agent
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agents.prompts import ORCHESTRATOR_AGENT_PROMPT
from agents.product_search_agent import internal_search
from agents.web_analysis_agent import web_search

orchestrator_agent = create_agent(
    model=os.getenv('ORCHESTRATOR_MODEL'),
    tools=[internal_search, web_search],
    system_prompt=ORCHESTRATOR_AGENT_PROMPT
)
