from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from search.engine import HybridSearchEngine

search_engine = HybridSearchEngine()


@tool
def search_products_tool(
        query: str, min_price: float = None, max_price: float = None,
        min_rating: float = None, max_rating: float = None,
        min_reviews_num: int = None, max_reviews_num : int = None,
        is_bestseller: bool = None, min_bought_in_last_month: int = None, max_bought_in_last_month: int = None,
        num_results: int = 10):
    """
    Use this function to look for products.
    Price is in USD
    Rating value from 0 to 5. 0 means no rating.
    Bought in last month means how many of that product was sold in the last month.
    Input:
    - query: product descriptions
    - min_price (Optional) minimum product price
    - max_price: (Optional) maximum product price
    - min_rating: (Optional) minimum product rating
    - max_rating: (Optional) maximum product rating
    - min_reviews_num: (Optional) minimum number of product reviews
    - max_reviews_num: (Optional) maximum number of product reviews
    - is_bestseller: (Optional) the product is bestseller or not
    - min_bought_in_last_month: (Optional) minimum number of bought in last month
    - max_bought_in_last_month: (Optional) maximum number of bought in last month
    - num_results: (Optional) number of retrieved products, default: 10
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

    response = ""
    for item in results:
        response += (f"- "
                     f"{item['title']} | Category: {item['category']} | Price: ${item['price']} | Rating: {item['rating']} | Reviews count: {item['reviews']} | \n")
    return response


@tool
def web_analysis_tool(query: str):
    """
    Use this tool ONLY for:
    1. Analysis: Providing insights, market trends, or comparisons.
    2. Question Answering: Asking for specific details NOT in the product list (e.g., "release date", "weight", "detailed reviews").
    3. External Info: When the user asks "Why", "How", or about "Trends".

    Input:
    - query: A targeted search query for the web
    """
    print(f'[WEB SEARCH] {query}')
    search = DuckDuckGoSearchResults()
    return search.run(query)
