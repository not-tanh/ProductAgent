from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from pydantic import BaseModel, Field

from search.engine import HybridSearchEngine

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
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Dynamic dictionary of extra product details (specs, color, etc.)")


class WebResult(BaseModel):
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str


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
