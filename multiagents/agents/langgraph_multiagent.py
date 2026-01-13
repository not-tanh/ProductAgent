import json
import os
from typing import Any, Dict, List, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from multiagents.agents.contracts import (
    Plan,
    ProductAnalysisSQL,
    Scratchpad,
    SearchProductsArgs,
    TaskType,
    WebQueries,
)
from multiagents.agents.tools import Product, WebResult, product_analysis_tool, search_products_tool, web_analysis_tool, get_schema_duckdb
from multiagents.agents.prompts import PLANNER_PROMPT, WEB_SEARCH_PROMPT, PRODUCT_SEARCH_PROMPT, PRODUCT_ANALYSIS_PROMPT, FINAL_PROMPT

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    scratch: dict[str, Any]


async def plan_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return {"scratch": Scratchpad().model_dump()}

    planner_model = ChatOpenAI(model=os.getenv("ORCHESTRATOR_MODEL"), temperature=0)
    planner = planner_model.with_structured_output(Plan)

    plan: Plan
    try:
        plan = await planner.ainvoke(
            [SystemMessage(content=PLANNER_PROMPT)] + messages[-8:],
            config=config,
        )
    except ValidationError:
        # Fail-closed: no tool execution if planning output is invalid
        plan = Plan(tasks=[])
    except Exception:
        plan = Plan(tasks=[])

    scratch = Scratchpad(tasks=plan.tasks, index=0, artifacts={})
    return {"scratch": scratch.model_dump()}


def _route(state: State) -> str:
    scratch = Scratchpad.model_validate(state.get("scratch") or {})
    task = scratch.current()
    if task is None:
        return "final"
    if task.type == TaskType.product_search:
        return "product_search"
    if task.type == TaskType.product_analysis:
        return "product_analysis"
    if task.type == TaskType.web_analysis:
        return "web_analysis"
    return "final"


async def dispatch_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    # No-op. We only use this node as a convenient place for conditional routing.
    return {}


async def product_search_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    scratch = Scratchpad.model_validate(state.get("scratch") or {})
    task = scratch.current()
    if task is None:
        return {}

    model = ChatOpenAI(model=os.getenv("PRODUCT_SEARCH_MODEL"), temperature=0)
    extractor = model.with_structured_output(SearchProductsArgs)

    args: SearchProductsArgs
    try:
        args = await extractor.ainvoke(
            [SystemMessage(content=PRODUCT_SEARCH_PROMPT), HumanMessage(content=task.request)],
            config=config,
        )
    except Exception:
        # Fail-closed: store an error artifact and continue
        scratch.artifacts[task.task_id] = {"type": "product_search", "ok": False, "error": "argument_extraction_failed"}
        scratch.index += 1
        return {"scratch": scratch.model_dump()}

    # Tool call (validated by tool schema + our contract)
    tool_input = args.model_dump(exclude_none=True)
    products_raw = await search_products_tool.ainvoke(tool_input, config=config)

    # Validate output contract (ensure serializable + consistent fields)
    products: List[Dict[str, Any]] = []
    for item in products_raw or []:
        try:
            products.append(Product.model_validate(item).model_dump())
        except Exception:
            # tolerate a single bad item
            continue

    scratch.artifacts[task.task_id] = {
        "type": "product_search",
        "ok": True,
        "args": tool_input,
        "products": products,
    }
    scratch.index += 1
    return {"scratch": scratch.model_dump()}


async def web_analysis_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    scratch = Scratchpad.model_validate(state.get("scratch") or {})
    task = scratch.current()
    if task is None:
        return {}

    model = ChatOpenAI(model=os.getenv("WEB_ANALYSIS_MODEL"), temperature=0)
    extractor = model.with_structured_output(WebQueries)

    try:
        q = await extractor.ainvoke(
            [SystemMessage(content=WEB_SEARCH_PROMPT), HumanMessage(content=task.request)],
            config=config,
        )
        queries = [s.strip() for s in q.queries if s and s.strip()][:3]
        if not queries:
            queries = [task.request]
    except Exception:
        queries = [task.request]

    results: List[Dict[str, Any]] = []
    for query in queries:
        out = await web_analysis_tool.ainvoke({"query": query}, config=config)
        for item in out or []:
            try:
                results.append(WebResult.model_validate(item).model_dump())
            except Exception:
                continue

    scratch.artifacts[task.task_id] = {
        "type": "web_analysis",
        "ok": True,
        "queries": queries,
        "results": results,
    }
    scratch.index += 1
    return {"scratch": scratch.model_dump()}


async def product_analysis_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    scratch = Scratchpad.model_validate(state.get("scratch") or {})
    task = scratch.current()
    if task is None:
        return {}

    model = ChatOpenAI(model=os.getenv("PRODUCT_ANALYSIS_MODEL"), temperature=0)
    extractor = model.with_structured_output(ProductAnalysisSQL)

    try:
        sql = await extractor.ainvoke(
            [SystemMessage(content=PRODUCT_ANALYSIS_PROMPT.format(schema=get_schema_duckdb())),
             HumanMessage(content=task.request)],
            config=config,
        )
        sql_query = sql.sql_query.strip()
    except Exception:
        sql_query = ""

    if not sql_query:
        scratch.artifacts[task.task_id] = {
            "type": "product_analysis", "ok": False, "error": "sql_generation_failed"}
        scratch.index += 1
        return {"scratch": scratch.model_dump()}

    out = await product_analysis_tool.ainvoke({"sql_query": sql_query}, config=config)

    scratch.artifacts[task.task_id] = {
        "type": "product_analysis",
        **(out if isinstance(out, dict) else {"ok": False, "error": "invalid_tool_output"}),
        "sql_query": sql_query,
    }
    scratch.index += 1
    return {"scratch": scratch.model_dump()}


async def final_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    scratch = Scratchpad.model_validate(state.get("scratch") or {})

    # Build a compact context blob for the LLM
    artifacts_json = json.dumps(scratch.artifacts, ensure_ascii=False)

    model = ChatOpenAI(model=os.getenv("ORCHESTRATOR_MODEL"), temperature=0)

    final_messages: List[BaseMessage] = [
        SystemMessage(content=FINAL_PROMPT),
        SystemMessage(content=f"ARTIFACTS_JSON:\n{artifacts_json}"),
    ] + messages[-8:]

    ai = await model.ainvoke(final_messages, config=config)

    # Clear scratchpad so it doesn't grow unbounded across turns
    return {"messages": [ai], "scratch": Scratchpad().model_dump()}


def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("plan", plan_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("product_search", product_search_node)
    workflow.add_node("web_analysis", web_analysis_node)
    workflow.add_node("product_analysis", product_analysis_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "dispatch")

    workflow.add_conditional_edges(
        "dispatch",
        _route,
        {
            "product_search": "product_search",
            "web_analysis": "web_analysis",
            "product_analysis": "product_analysis",
            "final": "final",
        },
    )

    # Loop back after each specialist step
    workflow.add_edge("product_search", "dispatch")
    workflow.add_edge("web_analysis", "dispatch")
    workflow.add_edge("product_analysis", "dispatch")

    workflow.add_edge("final", END)

    return workflow.compile()
