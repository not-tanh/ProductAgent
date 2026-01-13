import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from multiagents.agents.langgraph_multiagent import build_graph

planner_agent = build_graph()
