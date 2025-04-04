import httpx
import os
import asyncio
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Union, Callable
from tenacity import retry, stop_after_attempt, wait_exponential
import pathlib
from models import TokenInfo, ExchangeInfo, CategoryInfo
from tqdm.auto import tqdm
from tmai_api import TokenMetricsClient

# Base URL for Token Metrics API
BASE_URL = "https://api.tokenmetrics.com/v2"

# Cache directory
CACHE_DIR = pathlib.Path("lists")

def get_cache_path(cache_name: str) -> pathlib.Path:
    """
    Get the full path to a cached file
    
    Args:
        cache_name: Name of the cache file (without .csv extension)
        
    Returns:
        Path to the cache file
    """
    return CACHE_DIR / f"{cache_name}.csv"

def is_cache_valid(cache_name: str, max_age_hours: int = 24) -> bool:
    """
    Check if a cache file exists and is not too old
    
    Args:
        cache_name: Name of the cache file (without .csv extension)
        max_age_hours: Maximum age of the cache in hours
        
    Returns:
        True if cache is valid, False otherwise
    """
    cache_path = get_cache_path(cache_name)
    
    # Check if file exists
    if not cache_path.exists():
        return False
    
    # Check if file is not too old
    if max_age_hours > 0:
        file_age = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime))
        if file_age.total_seconds() > max_age_hours * 3600:
            return False
    
    return True

# Parse string representations of JSON in dataframes
def parse_json_strings(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse string representations of JSON in a dictionary record
    
    Args:
        record: Dictionary record from DataFrame
        
    Returns:
        Record with JSON strings parsed
    """
    result = {}
    for key, value in record.items():
        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                result[key] = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = value
    return result

# Async httpx client with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_from_api(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch data from Token Metrics API with retry logic.
    
    Args:
        endpoint: API endpoint path
        params: Query parameters
        
    Returns:
        API response as dictionary
    """
    url = f"{BASE_URL}/{endpoint}"
    headers = {
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()


async def fetch_all_pages(endpoint: str, base_params: Optional[Dict[str, Any]] = None, 
                        limit: int = 10000, max_pages: Optional[int] = None,
                        data_extractor: Optional[Callable] = None, 
                        desc: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all items from an API endpoint, using one request if possible
    
    Args:
        endpoint: API endpoint path
        base_params: Base query parameters
        limit: Number of items per request (use large value to minimize requests)
        max_pages: Maximum number of pages to fetch (None for all)
        data_extractor: Function to extract data from response, defaults to response.get("data", [])
        desc: Description for the progress bar
        
    Returns:
        Combined list of all items from all pages
    """
    if base_params is None:
        base_params = {}
    
    # Create a copy of base_params to avoid modifying the original
    params = base_params.copy()
    params["limit"] = limit
    
    # First try to get all items in one request with a large limit
    print(f"Attempting to fetch all items in one request (limit={limit})...")
    
    try:
        response = await fetch_from_api(endpoint, params)
        items = response.get("data", []) if not data_extractor else data_extractor(response)
        
        # Get total count if available
        total_count = response.get("meta", {}).get("total", 0)
        
        if total_count:
            print(f"API reports {total_count} total items available")
            
            # If we got all items in one request, we're done
            if len(items) >= total_count:
                print(f"✅ Successfully fetched all {len(items)} items in one request")
                return items
            
            # If we got less than the total but more than the limit,
            # the API might have given us all it has despite reporting more
            if len(items) >= limit:
                print(f"⚠️ Received {len(items)} items which is the maximum allowed per request")
                print(f"   API reports {total_count} total items but may not actually have that many")
                return items
                
            print(f"First request returned {len(items)} items, need to fetch more...")
        else:
            # If we got a reasonable number of items and no total count is reported,
            # we might have all the data
            if len(items) < limit:
                print(f"✅ Received {len(items)} items, which is less than the limit ({limit})")
                print(f"   Assuming we have all the data since no total count is reported")
                return items
            
            print(f"First request returned {len(items)} items, may need to fetch more...")
            
        # If we reached here, we need multiple requests
        all_items = items.copy()
        pbar_desc = desc or f"Fetching {endpoint}"
        
        # Estimate how many more requests we might need
        estimated_total = total_count if total_count > 0 else len(items) * 2
        estimated_pages = (estimated_total + limit - 1) // limit
        
        # Limit to max_pages if specified
        if max_pages is not None:
            estimated_pages = min(estimated_pages, max_pages)
            
        # Set up a progress bar for additional requests
        with tqdm(total=estimated_pages, desc=pbar_desc, unit="request") as pbar:
            # Update progress for the first request we already made
            pbar.update(1)
            
            # If we didn't get a full page of items, we might be done
            if len(items) < limit:
                pbar.write("Received fewer items than requested limit, assuming we have all data")
                return all_items
            
            # Continue fetching more pages if needed
            page = 1  # Start from page 1 (we already got page 0/first page)
            
            while True:
                # Break if we've reached max_pages
                if max_pages is not None and page >= max_pages:
                    break
                    
                # Update params for next page
                params = base_params.copy()
                params["limit"] = limit
                params["page"] = page
                
                try:
                    response = await fetch_from_api(endpoint, params)
                    items = response.get("data", []) if not data_extractor else data_extractor(response)
                    
                    # If no items returned, we've reached the end
                    if not items:
                        pbar.write("No more items returned, finished fetching")
                        break
                        
                    # Add new items
                    all_items.extend(items)
                    
                    # Update progress description
                    pbar.set_postfix({"total": len(all_items)})
                    
                    # If we got fewer items than requested, we've reached the end
                    if len(items) < limit:
                        pbar.write(f"Received {len(items)} items (< {limit}), finished fetching")
                        break
                        
                    # Move to next page and update progress
                    page += 1
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.write(f"Error fetching page {page}: {e}")
                    break
                
                # Add a small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
        return all_items
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


# Raw fetch functions for different endpoints
async def fetch_coins_list(limit: int = 10000, use_cache: bool = True, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch the raw coins list from the API or cache
    
    Args:
        limit: Maximum number of coins per request
        use_cache: Whether to use cached data if available
        max_pages: Maximum number of pages to fetch (None for all)
        
    Returns:
        List of coin data dictionaries
    """
    cache_name = "coins"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            # Convert DataFrame rows to dictionaries and parse JSON strings
            print(f"Loading {len(df)} coins from cache")
            records = df.to_dict(orient="records")
            return [parse_json_strings(record) for record in records]
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    # Fetch all pages from API if cache not used or invalid
    return await fetch_all_pages("coins", limit=limit, max_pages=max_pages, desc="Fetching coins")


async def fetch_categories_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch the raw categories list from the API or cache
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        List of category data
    """
    cache_name = "categories"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            print(f"Loading {len(df)} categories from cache")
            # Convert DataFrame rows to dictionaries
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    try:
        # Fetch all pages of categories
        return await fetch_all_pages("categories", desc="Fetching categories")
    except Exception as e:
        print(f"Error fetching categories from API: {e}")
        print("Extracting categories from coins...")
        
        # If endpoint doesn't exist, extract unique categories from coins
        coins = await fetch_coins_list(use_cache=use_cache)
        categories = {}
        
        for coin in coins:
            if coin.get("CATEGORY_LIST"):
                for category in coin["CATEGORY_LIST"]:
                    key = category.get("category_id")
                    if key and key not in categories:
                        categories[key] = category
        
        return list(categories.values())


async def fetch_exchanges_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch the raw exchanges list from the API or cache
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        List of exchange data
    """
    cache_name = "exchanges"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            print(f"Loading {len(df)} exchanges from cache")
            # Convert DataFrame rows to dictionaries
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    try:
        # Fetch all pages of exchanges
        return await fetch_all_pages("exchanges", desc="Fetching exchanges")
    except Exception as e:
        print(f"Error fetching exchanges from API: {e}")
        print("Extracting exchanges from coins...")
        
        # If endpoint doesn't exist, extract unique exchanges from coins
        coins = await fetch_coins_list(use_cache=use_cache)
        exchanges = {}
        
        for coin in coins:
            if coin.get("EXCHANGE_LIST"):
                for exchange in coin["EXCHANGE_LIST"]:
                    key = exchange.get("exchange_id")
                    if key and key not in exchanges:
                        exchanges[key] = exchange
        
        return list(exchanges.values())


async def fetch_blockchains_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch the raw blockchains list from the API or cache
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        List of blockchain data
    """
    cache_name = "blockchains"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            print(f"Loading {len(df)} blockchains from cache")
            # Convert DataFrame rows to dictionaries
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    # Fetch all pages of blockchains
    return await fetch_all_pages("blockchains", desc="Fetching blockchains")


async def fetch_technical_indicators_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch the raw technical indicators list from the API or cache
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        List of technical indicator data
    """
    cache_name = "technical_indicators"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            print(f"Loading {len(df)} technical indicators from cache")
            # Convert DataFrame rows to dictionaries
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    # Fetch all pages of technical indicators
    return await fetch_all_pages("technical-indicators", desc="Fetching technical indicators")


async def fetch_ai_report_tokens_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch the list of all tokens for AI reports from the API or cache
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        List of AI report token data
    """
    cache_name = "ai_report_tokens"
    
    # Try to use cache if requested
    if use_cache and is_cache_valid(cache_name):
        try:
            # Read from cache
            df = pd.read_csv(get_cache_path(cache_name))
            print(f"Loading {len(df)} AI report tokens from cache")
            # Convert DataFrame rows to dictionaries
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading from cache: {e}")
    
    # Fetch all pages of AI report tokens
    return await fetch_all_pages("ai-reports-tokens", desc="Fetching AI report tokens")


async def get_coin_data(symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific coin by symbol.
    Uses the cached /v2/coins list by default.
    
    Args:
        symbol: Coin symbol (e.g., BTC, ETH)
        use_cache: Whether to use cached coins list data if available
        
    Returns:
        Detailed coin information if found, None otherwise.
    """
    print(f"Attempting to get token data for {symbol} from coins list (use_cache={use_cache})")
    try:
        # This uses the potentially broken /v2/coins endpoint via fetch_coins_list
        coins = await fetch_coins_list(use_cache=use_cache)
        if not coins:
            print("Warning: fetch_coins_list returned empty result.")
            return None
            
        for coin in coins:
            # Case-insensitive comparison for robustness
            if coin.get("TOKEN_SYMBOL", "").upper() == symbol.upper():
                print(f"Found {symbol} in coins list.")
                return coin
                
        print(f"Symbol {symbol} not found in coins list.")
        return None
    except Exception as e:
        print(f"Error fetching or processing coins list for {symbol}: {e}")
        # This might happen if the /v2/coins endpoint call inside fetch_coins_list fails
        return None

async def get_token_data_via_tokens_endpoint(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific token using the premium /v2/tokens endpoint.
    This serves as a fallback if the /v2/coins endpoint is unavailable.
    Requires API key.
    
    Args:
        symbol: Token symbol (e.g., BTC, ETH)
        
    Returns:
        Detailed token information if found, None otherwise.
    """
    print(f"Falling back to fetching token data for {symbol} using /v2/tokens endpoint.")
    
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        print("Error: TOKEN_METRICS_API_KEY not set. Cannot use /v2/tokens endpoint.")
        return None

    try:
        client = TokenMetricsClient(api_key=api_key)
        # Use await for the async call
        response = await asyncio.to_thread(client.tokens.get, symbol=symbol)
        
        if response and response.get("data"):
            token_data_list = response["data"]
            if token_data_list:
                token_data = token_data_list[0] # Assume first is correct
                if token_data.get("TOKEN_SYMBOL", "").upper() == symbol.upper():
                    print(f"Successfully fetched data for {symbol} via /v2/tokens")
                    return token_data
                else:
                    print(f"Warning: /v2/tokens fetched data symbol '{token_data.get('TOKEN_SYMBOL')}' doesn't match requested '{symbol}'")
                    # Decide if mismatch is acceptable or return None
                    # Returning None for safety for now
                    return None 
            else:
                print(f"No data list found for symbol {symbol} in the /v2/tokens response.")
                return None
        else:
            print(f"Failed to fetch or no data returned for symbol {symbol} from /v2/tokens.")
            return None
            
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        print(f"Error fetching token data for {symbol} via /v2/tokens: {e}")
        traceback.print_exc() # Print detailed traceback
        return None

async def cache_lists(force_refresh: bool = False) -> None:
    """
    Fetch all list data from TokenMetrics API, print lengths, and save to CSV files.
    Creates a 'lists' directory if it doesn't exist and saves each list as a CSV.
    
    Args:
        force_refresh: Whether to force a refresh of the cache
    """
    # Create lists directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    try:
        # Fetch coins list - this is the most important one
        print("Fetching all coins (this may take a while)...")
        coins_data = await fetch_coins_list(use_cache=not force_refresh)
        print(f"Coins list length: {len(coins_data)}")
        coins_df = pd.DataFrame(coins_data)
        coins_df.to_csv(get_cache_path("coins"), index=False)
        print(f"Saved coins list to {get_cache_path('coins')}")
        
        # Fetch exchanges
        try:
            print("Fetching all exchanges...")
            exchanges = await fetch_exchanges_list(use_cache=not force_refresh)
            print(f"Exchanges list length: {len(exchanges)}")
            exchanges_df = pd.DataFrame(exchanges)
            exchanges_df.to_csv(get_cache_path("exchanges"), index=False)
            print(f"Saved exchanges list to {get_cache_path('exchanges')}")
        except Exception as e:
            print(f"Error fetching exchanges list: {e}")
            
        # Fetch categories
        try:
            print("Fetching all categories...")
            categories = await fetch_categories_list(use_cache=not force_refresh)
            print(f"Categories list length: {len(categories)}")
            categories_df = pd.DataFrame(categories)
            categories_df.to_csv(get_cache_path("categories"), index=False)
            print(f"Saved categories list to {get_cache_path('categories')}")
        except Exception as e:
            print(f"Error fetching categories list: {e}")
        
        # Fetch blockchains
        try:
            print("Fetching all blockchains...")
            blockchains = await fetch_blockchains_list(use_cache=not force_refresh)
            print(f"Blockchains list length: {len(blockchains)}")
            blockchains_df = pd.DataFrame(blockchains)
            blockchains_df.to_csv(get_cache_path("blockchains"), index=False)
            print(f"Saved blockchains list to {get_cache_path('blockchains')}")
        except Exception as e:
            print(f"Error fetching blockchains list: {e}")
        
        # Fetch technical indicators
        try:
            print("Fetching all technical indicators...")
            indicators = await fetch_technical_indicators_list(use_cache=not force_refresh)
            print(f"Technical indicators list length: {len(indicators)}")
            indicators_df = pd.DataFrame(indicators)
            indicators_df.to_csv(get_cache_path("technical_indicators"), index=False)
            print(f"Saved technical indicators list to {get_cache_path('technical_indicators')}")
        except Exception as e:
            print(f"Error fetching technical indicators list: {e}")
        
        # Fetch AI report tokens
        try:
            print("Fetching all AI report tokens...")
            ai_tokens = await fetch_ai_report_tokens_list(use_cache=not force_refresh)
            print(f"AI report tokens list length: {len(ai_tokens)}")
            ai_tokens_df = pd.DataFrame(ai_tokens)
            ai_tokens_df.to_csv(get_cache_path("ai_report_tokens"), index=False)
            print(f"Saved AI report tokens list to {get_cache_path('ai_report_tokens')}")
        except Exception as e:
            print(f"Error fetching AI report tokens list: {e}")
            
    except Exception as e:
        print(f"Error fetching coin data: {e}")
    
    print(f"Lists cached to {CACHE_DIR} directory")


async def invalidate_cache() -> None:
    """
    Invalidate and refresh the cache by forcing a refresh of all lists
    """
    print("Invalidating and refreshing cache...")
    await cache_lists(force_refresh=True)
    print("Cache refresh complete") 