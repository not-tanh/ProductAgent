import os
import sys
from typing import List, Optional, Any

import duckdb
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.engine import HybridSearchEngine
load_dotenv()

search_engine = HybridSearchEngine()


class Product(BaseModel):
    """Represents a single product result with standardized fields."""
    title: str = Field(description="Name or title of the product")
    category: str = Field(description="Product category")
    price: float = Field(description="Price in USD")
    rating: float = Field(description="Average rating from 0 to 5", ge=0, le=5)
    reviews: int = Field(description="Total count of reviews")
    isBestSeller: bool = Field(description="Whether the product is flagged as a bestseller")
    boughtInLastMonth: int = Field(description="Number of units sold in the last month")


class WebResult(BaseModel):
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str


class QueryResult(BaseModel):
    """Structured SQL result for durable, JSON-serializable agent execution."""
    ok: bool
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None


@tool
def search_products_tool(
        query: str, min_price: float = None, max_price: float = None,
        min_rating: float = None, max_rating: float = None,
        min_reviews_num: int = None, max_reviews_num : int = None,
        is_bestseller: bool = None, min_bought_in_last_month: int = None, max_bought_in_last_month: int = None,
        num_results: int = 10) -> List[Product]:
    """
    Use this function to look for products.
    Price is in USD
    Rating value from 0 to 5. 0 means no rating.

    Input:
    - query: product descriptions
    - min/max price, rating, reviews, bought_last_month: Filters
    - is_bestseller: Filter for bestsellers
    - num_results: Number of retrieved products (default 10)
    """

    results = search_engine.search(
        query_text=query,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        max_rating=max_rating,
        min_reviews_num=min_reviews_num,
        max_reviews_num=max_reviews_num,
        min_bought_in_last_month=min_bought_in_last_month,
        max_bought_in_last_month=max_bought_in_last_month,
        is_bestseller=is_bestseller,
        top_k=num_results
    )

    structured_products = [Product(**item) for item in results]
    return structured_products


@tool
def web_analysis_tool(query: str) -> List[WebResult]:
    """
    Use this tool ONLY for:
    1. Analysis: Providing insights, market trends, or comparisons.
    2. Question Answering: Asking for specific details NOT in the product list (e.g., "release date", "weight", "detailed reviews").
    3. External Info: When the user asks "Why", "How", or about "Trends".

    Input:
    - query: A targeted search query for the web
    """

    search = DuckDuckGoSearchAPIWrapper()
    results = search.results(query, max_results=5)

    structured_response = []
    for item in results:
        structured_response.append(WebResult(
            title=item.get("title", "No Title"),
            url=item.get("link", ""),
            snippet=item.get("snippet", "")
        ))

    return structured_response


@tool
def product_analysis_tool(sql_query: str) -> QueryResult:
    """
    Executes a read-only SQL query and returns structured results.

    This is intentionally JSON-serializable to support durable checkpointing
    and contract validation across agents.
    """
    conn = None
    try:
        conn = duckdb.connect(os.getenv('DUCKDB'), read_only=True)

        forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
        if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
            return QueryResult(ok=False, error="Read-only access. Modification commands are not allowed.")

        cursor = conn.execute(sql_query)

        # Limit the query to prevent context-window bloat
        rows = cursor.fetchmany(25)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        return QueryResult(ok=True, columns=columns, rows=[list(r) for r in rows])

    except Exception as e:
        # Return the actual database error so the LLM can self-correct
        return QueryResult(ok=False, error=f"{type(e).__name__}: {str(e)}")
    finally:
        if conn:
            conn.close()


def get_schema_duckdb():
        """
        Returns the table schema so the LLM knows which columns to query.
        """
        conn = None
        try:
            conn = duckdb.connect(os.getenv('DUCKDB'), read_only=True)
            schema_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'products'
            """
            results = conn.execute(schema_query).fetchall()
            # Format as a clean string for the LLM
            schema_str = "Table 'products' columns:\n"
            for col, dtype in results:
                schema_str += f"- {col} ({dtype})\n"
            return schema_str
        except Exception as e:
            return f"Error fetching schema: {str(e)}"
        finally:
            if conn:
                conn.close()
