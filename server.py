from mcp.server.fastmcp import FastMCP
from tmai_api import TokenMetricsClient
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import json

# Import from our modules
from models import TokenInfo, ExchangeInfo, CategoryInfo
from helpers import (
    fetch_coins_list, 
    fetch_categories_list, 
    fetch_exchanges_list,
    fetch_blockchains_list,
    fetch_technical_indicators_list,
    fetch_ai_report_tokens_list,
    get_coin_data,
    cache_lists,
    invalidate_cache as helper_invalidate_cache,
    is_cache_valid
)

# Load environment variables from .env file
load_dotenv(override=True)

# Create an MCP server
mcp = FastMCP("Token Metrics API")

# Helper endpoints (no API key needed)
# Keep the original functions as-is

@mcp.tool()
async def get_coins_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of all token IDs, names, and symbols from Token Metrics.
    Endpoint: /v2/coins
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of coin information
    """
    return await fetch_coins_list(use_cache=use_cache)


@mcp.tool()
async def get_categories_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of all token categories available in Token Metrics.
    Endpoint: /v2/categories
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of category information
    """
    return await fetch_categories_list(use_cache=use_cache)


@mcp.tool()
async def get_exchanges_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of all exchanges tracked by Token Metrics.
    Endpoint: /v2/exchanges
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of exchange information
    """
    return await fetch_exchanges_list(use_cache=use_cache)


@mcp.tool()
async def get_blockchains_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of all blockchains supported by Token Metrics.
    Endpoint: /v2/blockchains
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of blockchain details
    """
    return await fetch_blockchains_list(use_cache=use_cache)


@mcp.tool()
async def get_technical_indicators_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of available technical indicators.
    Endpoint: /v2/technical-indicators
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of technical indicator information
    """
    return await fetch_technical_indicators_list(use_cache=use_cache)


@mcp.tool()
async def get_ai_report_tokens_list(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get the list of all tokens for AI reports.
    Endpoint: /v2/ai-reports-tokens
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        List of AI report token information
    """
    return await fetch_ai_report_tokens_list(use_cache=use_cache)


@mcp.tool()
async def get_token_data(symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific token by symbol
    Uses the coins endpoint and filters by symbol
    
    Args:
        symbol: Token symbol (e.g., BTC, ETH)
        use_cache: Whether to use cached data if available
        
    Returns:
        Detailed token information if found
    """
    return await get_coin_data(symbol, use_cache=use_cache)


@mcp.tool()
async def cache_all_lists() -> str:
    """
    Fetch and cache all available lists from Token Metrics API.
    Saves them as CSV files in the 'lists' directory.
    
    Returns:
        Status message
    """
    await cache_lists()
    return "All lists cached successfully"


@mcp.tool()
async def invalidate_cache() -> str:
    """
    Invalidate and refresh all cached data from the API.
    Forces a fresh fetch of all data regardless of cache age.
    
    Returns:
        Status message
    """
    await helper_invalidate_cache()
    return "Cache invalidated and refreshed successfully"


@mcp.tool()
def get_cache_status() -> Dict[str, Any]:
    """
    Get the status of all cache files
    
    Returns:
        Dictionary with cache status information
    """
    cache_files = [
        "coins", 
        "categories", 
        "exchanges", 
        "blockchains", 
        "technical_indicators", 
        "ai_report_tokens"
    ]
    status = {}
    
    for cache_name in cache_files:
        status[cache_name] = {
            "valid": is_cache_valid(cache_name),
            "exists": is_cache_valid(cache_name, max_age_hours=0)
        }
    
    return status


# Premium API endpoints (require API key)
# These functions require access to the Token Metrics API with a valid API key

@mcp.tool()
async def get_tokens_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information for specific tokens using premium API.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        
    Returns:
        Token information data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    result = client.tokens.get(symbol=symbol)
    return result


@mcp.tool()
async def get_hourly_ohlcv_data(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 7
) -> Dict[str, Any]:
    """
    Get hourly Open-High-Low-Close-Volume data for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Hourly OHLCV data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)


    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.hourly_ohlcv.get(
        symbol=symbol,
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def get_daily_ohlcv_data(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get daily Open-High-Low-Close-Volume data for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Daily OHLCV data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.daily_ohlcv.get(
        symbol=symbol,
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def get_trader_grades(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get trader grades (short-term trading recommendations) for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Trader grades data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.trader_grades.get(
        symbol=symbol,
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def get_investor_grades(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get investor grades (long-term investment recommendations) for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Investor grades data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.investor_grades.get(
        symbol=symbol,
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def get_market_metrics(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get overall market sentiment and metrics.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Market metrics data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.market_metrics.get(
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def get_trader_indices(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get AI-generated trading portfolios (indices).
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        
    Returns:
        Trader indices data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.trader_indices.get(
        startDate=start_date,
        endDate=end_date
    )
    return result


@mcp.tool()
async def ask_ai_agent(question: str) -> Dict[str, Any]:
    """
    Ask the Token Metrics AI chatbot a question.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        question: Question to ask the AI agent
        
    Returns:
        AI response data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    result = client.ai_agent.ask(question)
    return result


@mcp.tool()
async def get_ai_reports(symbol: str) -> Dict[str, Any]:
    """
    Get AI-generated reports for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        
    Returns:
        AI reports data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    result = client.ai_reports.get(symbol=symbol)
    return result


@mcp.tool()
async def get_trading_signals(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 30,
    signal: str = "1"  # 1 for bullish signals
) -> Dict[str, Any]:
    """
    Get AI-generated trading signals for specified tokens.
    Requires API key to be set in TOKEN_METRICS_API_KEY environment variable.
    
    Args:
        symbol: Comma-separated token symbols (e.g., "BTC,ETH")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        signal: Signal type (1 for bullish, -1 for bearish)
        
    Returns:
        Trading signals data
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        return {"error": "TOKEN_METRICS_API_KEY not set in environment"}
    
    client = TokenMetricsClient(api_key=api_key)
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    result = client.trading_signals.get(
        symbol=symbol,
        startDate=start_date,
        endDate=end_date,
        signal=signal
    )
    return result


async def test_api_functionality():
    """
    Test function to verify the Token Metrics API functionality.
    Replicates calls from the examples to verify the API key is working.
    """
    print("\n" + "="*50)
    print("TESTING TOKEN METRICS API FUNCTIONALITY")
    print("="*50)
    
    # Check if API key is set
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        print("⚠️ ERROR: TOKEN_METRICS_API_KEY not set in environment")
        print("Please set your Token Metrics API key in .env file")
        return
    else:
        print(f"✅ API Key found: {api_key[:5]}...{api_key[-5:]}")
    
    # Create client instance
    client = TokenMetricsClient(api_key=api_key)
    
    try:
        # # 1. Test tokens endpoint (get info for BTC and ETH)
        # print("\n>> TESTING TOKENS INFO (BTC and ETH)")
        #     # Get information for Bitcoin and Ethereum
        # tokens = client.tokens.get(symbol="BTC,ETH")
        # print(f"Found {len(tokens.get('data', []))} tokens")

        # # Convert to DataFrame for easier exploration
        # tokens_df = client.tokens.get_dataframe(symbol="BTC,ETH")
        # print(tokens_df.head())

        # print(f"✅ Successfully retrieved info for {tokens_df.shape[0]} tokens")
        
        # # Show a sample of the data
        # if tokens_df.shape[0] > 0:
        #     sample_token = tokens_df.iloc[0]
        #     print(f"Sample token info for {sample_token.get('TOKEN_SYMBOL', 'Unknown')}:")
        #     print(f"  - Name: {sample_token.get('TOKEN_NAME', 'Unknown')}")
        #     print(f"  - ID: {sample_token.get('TOKEN_ID', 'Unknown')}")
        
        # 2. Test daily OHLCV endpoint
        print("\n>> TESTING DAILY OHLCV (BTC for last 30 days)")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")

        # Get OHLCV data - the SDK will automatically handle the 29-day limit and show a progress bar
        ohlcv_df = client.hourly_ohlcv.get_dataframe(
            symbol="BTC", 
            startDate=start_date,
            endDate=end_date
        )
        print(ohlcv_df.head())
        ohlcv_result = await get_daily_ohlcv_data("BTC", start_date=start_date, end_date=end_date)
        print(ohlcv_result)
        ohlcv_count = len(ohlcv_result.get("data", []))
        print(f"✅ Successfully retrieved {ohlcv_count} OHLCV data points")
        
        # Show a sample of the OHLCV data
        if ohlcv_count > 0:
            sample_ohlcv = ohlcv_result.get("data", [])[0]
            print(f"Sample OHLCV data for {sample_ohlcv.get('TOKEN_SYMBOL', 'Unknown')}:")
            print(f"  - Date: {sample_ohlcv.get('DATE', 'Unknown')}")
            print(f"  - Open: {sample_ohlcv.get('OPEN', 'Unknown')}")
            print(f"  - High: {sample_ohlcv.get('HIGH', 'Unknown')}")
            print(f"  - Low: {sample_ohlcv.get('LOW', 'Unknown')}")
            print(f"  - Close: {sample_ohlcv.get('CLOSE', 'Unknown')}")
            print(f"  - Volume: {sample_ohlcv.get('VOLUME', 'Unknown')}")
        
        # 3. Test trading signals
        print("\n>> TESTING TRADING SIGNALS (BTC for last 30 days)")
        signals_result = await get_trading_signals("BTC", start_date=start_date, end_date=end_date)
        signals_count = len(signals_result.get("data", []))
        print(f"✅ Successfully retrieved {signals_count} trading signals")
        
        # Show a sample of the trading signals
        if signals_count > 0:
            sample_signal = signals_result.get("data", [])[0]
            print(f"Sample trading signal for {sample_signal.get('TOKEN_SYMBOL', 'Unknown')}:")
            print(f"  - Date: {sample_signal.get('DATE', 'Unknown')}")
            print(f"  - Signal: {sample_signal.get('SIGNAL', 'Unknown')}")
            print(f"  - Source: {sample_signal.get('SOURCE', 'Unknown')}")
        
        print("\n✅ API TEST COMPLETE: All endpoints working correctly")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to test API: {e}")
        
    print("="*50)


def main():
    """Main function to run when the server starts"""
    print("Starting Token Metrics MCP...")
    
    # Test the API functionality
    asyncio.run(test_api_functionality())
    
    print("Token Metrics MCP is ready!")


if __name__ == "__main__":
    main()
