import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    product_search = "product_search"
    product_analysis = "product_analysis"
    web_analysis = "web_analysis"


class AgentTask(BaseModel):
    """Contract from planner -> executor nodes."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    request: str = Field(
        description="Natural-language request for the specialist agent."
    )


class Plan(BaseModel):
    """Contract produced by the planner node."""

    tasks: list[AgentTask] = Field(default_factory=list)


class Scratchpad(BaseModel):
    """Durable per-turn working memory stored in graph state."""

    tasks: list[AgentTask] = Field(default_factory=list)
    index: int = 0
    artifacts: dict[str, Any] = Field(default_factory=dict)

    def current(self) -> Optional[AgentTask]:
        if self.index < 0 or self.index >= len(self.tasks):
            return None
        return self.tasks[self.index]


class SearchProductsArgs(BaseModel):
    query: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_reviews_num: Optional[int] = None
    max_reviews_num: Optional[int] = None
    is_bestseller: Optional[bool] = None
    min_bought_in_last_month: Optional[int] = None
    max_bought_in_last_month: Optional[int] = None
    num_results: int = 10


class WebQueries(BaseModel):
    queries: list[str] = Field(
        description="One or more targeted web search queries."
    )


class ProductAnalysisSQL(BaseModel):
    sql_query: str = Field(description="Read-only DuckDB SQL over table `products`.")


class QueryResult(BaseModel):
    ok: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    error: Optional[str] = None
