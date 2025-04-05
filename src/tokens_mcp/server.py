from mcp.server.fastmcp import FastMCP
from tmai_api import TokenMetricsClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import vectorbt as vbt
import matplotlib.pyplot as plt
import base64
import io
from mcp.types import ImageContent
from eliot import start_action
from pathlib import Path


# Import from our modules
from tokens_mcp.models import EmaResponse, MACrossoverResponse, StrategyStats
from tokens_mcp.helpers import (
    fetch_coins_list, 
    fetch_categories_list, 
    fetch_exchanges_list,
    fetch_blockchains_list,
    fetch_technical_indicators_list,
    fetch_ai_report_tokens_list,
    cache_lists,
    invalidate_cache as helper_invalidate_cache,
    is_cache_valid,
    get_token_data_via_tokens_endpoint
)

# Load environment variables from .env file
load_dotenv(override=True)

# Create an MCP server
mcp = FastMCP("TmaiAPI")

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
    #return await get_coin_data(symbol, use_cache=use_cache)
    return await get_token_data_via_tokens_endpoint(symbol)

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


@mcp.tool()
async def get_emas(
    symbol: str, 
    timeframe: str = "1D", 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 100,
    ema_periods: List[int] = [10, 20, 50, 100, 200]
) -> EmaResponse:
    """
    Get Exponential Moving Averages (EMAs) for a specified token and timeframe using vectorbt.
    
    Args:
        symbol: Token symbol (e.g., "BTC", "ETH")
        timeframe: Timeframe for data ("1D" for daily, "1H" for hourly)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        ema_periods: List of EMA periods to calculate
        
    Returns:
        EmaResponse object containing price data and EMAs
    """
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        raise ValueError("TOKEN_METRICS_API_KEY not set in environment")
    
    client = TokenMetricsClient(api_key=api_key)
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get price data based on timeframe
    if timeframe.upper() == "1D":
        response = client.daily_ohlcv.get(
            symbol=symbol,
            startDate=start_date,
            endDate=end_date
        )
    elif timeframe.upper() == "1H":
        response = client.hourly_ohlcv.get(
            symbol=symbol,
            startDate=start_date,
            endDate=end_date
        )
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use '1D' or '1H'.")
    
    # Check if data was retrieved successfully
    if not response.get("data"):
        raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
    
    # Convert to DataFrame
    df = pd.DataFrame(response["data"])
    
    # Ensure datetime index
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate EMAs using vectorbt
    close_series = df["CLOSE"]
    ema_data = {}
    
    for period in ema_periods:
        # Calculate EMA using vectorbt
        ema = vbt.MA.run(close_series, period, short_name=f'EMA{period}', ewm=True)
        ema_data[f"EMA{period}"] = ema.ma.to_numpy().tolist()
    
    # Create and return EmaResponse
    return EmaResponse(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        dates=df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        close=df["CLOSE"].tolist(),
        open=df["OPEN"].tolist(),
        high=df["HIGH"].tolist(),
        low=df["LOW"].tolist(),
        volume=df["VOLUME"].tolist(),
        emas=ema_data
    )


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


@mcp.tool()
async def ma_crossover_strategy(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ma_type: str = "EMA",
    ma_periods: Tuple[int, int] = (20, 50),
    stop_loss_pct: Optional[float] = None,
) -> MACrossoverResponse:
    """
    Calculate moving average crossover strategy for a given token.
    
    Args:
        symbol: Token symbol (e.g., "BTC")
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
        ma_type: Moving average type ("SMA" or "EMA")
        ma_periods: Tuple of short and long MA periods
        stop_loss_pct: Optional trailing stop loss percentage
        
    Returns:
        MACrossoverResponse with strategy results
    """
    
    # Get daily OHLCV data
    with start_action(action_type="ma_crossover_strategy", symbol=symbol, 
                           start_date=start_date, end_date=end_date) as action:
        action.log(message_type="strategy_start", message=f"Running strategy for {symbol} from {start_date} to {end_date}")
        api_key = os.getenv("TOKEN_METRICS_API_KEY")
        if not api_key:
            raise ValueError("TOKEN_METRICS_API_KEY not set in environment")
        
        client = TokenMetricsClient(api_key=api_key)
        
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get the daily OHLCV data
        response = client.daily_ohlcv.get(
            symbol=symbol,
            startDate=start_date,
            endDate=end_date
        )
        
        # Check if data was retrieved successfully
        if not response.get("data"):
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        data = response["data"]
        
        # --- Filter data to only include the primary token for the given symbol ---
        # Get the expected token name for the symbol
        token_info = await get_token_data(symbol, use_cache=True) 
        
        # Fallback to /v2/tokens endpoint if /v2/coins failed
        if not token_info:
            action.log(message_type="api_fallback", message=f"get_token_data (using coins list) failed for {symbol}. Trying /v2/tokens endpoint...")
            token_info = await get_token_data_via_tokens_endpoint(symbol)
        
        if not token_info or 'TOKEN_NAME' not in token_info:
            raise ValueError(f"Could not retrieve token information for symbol {symbol} using either endpoint.")
            
        expected_token_name = token_info['TOKEN_NAME']
        action.log(message_type="token_info", message=f"Using TOKEN_NAME '{expected_token_name}' for filtering.")
        
        # Print sample of raw data for debugging
        with start_action(action_type="data_processing", phase="raw_data") as data_action:
            data_action.log(message_type="raw_data_sample", message="Sample Raw API Data")
            data_action.log(message_type="data_stats", data_count=str(len(data)))
            data_action.log(message_type="data_sample", first_items=str(data[:3]))
            data_action.log(message_type="data_sample", last_items=str(data[-3:]))
        
        # Filter the data
        filtered_data = [item for item in data if item.get('TOKEN_NAME') == expected_token_name]
        
        if not filtered_data:
            action.log(message_type="filter_warning", message=f"WARNING: No data remaining after filtering for TOKEN_NAME '{expected_token_name}'. Check API response.")
            # Optionally, fall back to using TOKEN_SYMBOL if name filtering fails?
            # For now, proceed with empty data which will likely error out later but more cleanly.
        
        with start_action(action_type="data_processing", phase="filtered_data") as filter_action:
            filter_action.log(message_type="filter_stats", message="Filtered Data Stats")
            filter_action.log(message_type="filter_counts", original_count=str(len(data)))
            filter_action.log(message_type="filter_counts", filtered_count=str(len(filtered_data)))
            filter_action.log(message_type="filter_counts", removed_count=str(len(data) - len(filtered_data)))
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        # Check if DataFrame is empty after filtering
        if df.empty:
             raise ValueError(f"No data available for the primary token '{expected_token_name}' ({symbol}) after filtering.")
             
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted chronologically
        
        # Convert OHLCV columns to numeric
        with start_action(action_type="data_processing", phase="convert_numeric") as convert_action:
            convert_action.log(message_type="conversion", message="Converting OHLCV columns to numeric...")
            for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Debug post-conversion data quality
            convert_action.log(message_type="data_quality", message="Post-Conversion DataFrame Checks")
            convert_action.log(message_type="data_quality", nan_count=str(int(df['CLOSE'].isna().sum())))
            
            # Look for suspiciously low close prices (for BTC, anything < 40k in 2024-25 is suspicious)
            low_price_rows = df[df['CLOSE'] < 40000]
            if len(low_price_rows) > 0:
                convert_action.log(message_type="data_quality_issue", message=f"Found {len(low_price_rows)} rows with close price < 40000")
                convert_action.log(message_type="data_quality_sample", low_price_rows=str(low_price_rows['CLOSE'].to_frame()))
            
            # Check for zero prices
            zero_price_rows = df[df['CLOSE'] == 0]
            if len(zero_price_rows) > 0:
                convert_action.log(message_type="data_quality_issue", message=f"Found {len(zero_price_rows)} rows with close price == 0")
                convert_action.log(message_type="data_quality_sample", zero_price_rows=str(zero_price_rows['CLOSE'].to_frame()))
        
        # Rename for clarity
        df = df.rename(columns={
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume'
        })
        
        # Create a deep copy for analysis to avoid any modification by vectorbt
        df_analysis = df[['close']].copy(deep=True)
        
        with start_action(action_type="data_processing", phase="analysis_prep") as prep_action:
            prep_action.log(message_type="data_prep", message="Data preparation stats")
            prep_action.log(message_type="data_prep", data_points=str(len(df)))
            prep_action.log(message_type="data_prep", date_range_start=str(df.index.min()))
            prep_action.log(message_type="data_prep", date_range_end=str(df.index.max()))
            prep_action.log(message_type="data_prep", df_analysis_shape=str(df_analysis.shape))
        
        # Calculate Moving Averages on the isolated dataframe
        if ma_type == "SMA":
            df_analysis[f'SMA{ma_periods[0]}'] = df_analysis['close'].rolling(window=ma_periods[0]).mean()
            df_analysis[f'SMA{ma_periods[1]}'] = df_analysis['close'].rolling(window=ma_periods[1]).mean()
        elif ma_type == "EMA":
            df_analysis[f'EMA{ma_periods[0]}'] = df_analysis['close'].ewm(span=ma_periods[0], adjust=False).mean()
            df_analysis[f'EMA{ma_periods[1]}'] = df_analysis['close'].ewm(span=ma_periods[1], adjust=False).mean()
        
        # Drop NaN values (due to MA calculation)
        df_clean = df_analysis.dropna()
        action.log(message_type="data_cleaning", message=f"After dropping NaN values: {len(df_clean)} data points")
        
        # Report MA value ranges
        with start_action(action_type="technical_indicators", phase="ma_values") as ma_action:
            if ma_type == "SMA":
                ma_action.log(message_type="sma_ranges", message="SMA value ranges")
                ma_action.log(message_type="sma_ranges", short_ma_min=str(float(df_clean[f'SMA{ma_periods[0]}'].min())))
                ma_action.log(message_type="sma_ranges", short_ma_max=str(float(df_clean[f'SMA{ma_periods[0]}'].max())))
                ma_action.log(message_type="sma_ranges", long_ma_min=str(float(df_clean[f'SMA{ma_periods[1]}'].min())))
                ma_action.log(message_type="sma_ranges", long_ma_max=str(float(df_clean[f'SMA{ma_periods[1]}'].max())))
            else:
                ma_action.log(message_type="ema_ranges", message="EMA value ranges")
                ma_action.log(message_type="ema_ranges", short_ma_min=str(float(df_clean[f'EMA{ma_periods[0]}'].min())))
                ma_action.log(message_type="ema_ranges", short_ma_max=str(float(df_clean[f'EMA{ma_periods[0]}'].max())))
                ma_action.log(message_type="ema_ranges", long_ma_min=str(float(df_clean[f'EMA{ma_periods[1]}'].min())))
                ma_action.log(message_type="ema_ranges", long_ma_max=str(float(df_clean[f'EMA{ma_periods[1]}'].max())))
        
        # Generate signals based on MA crossover
        short_ma = f"{ma_type}{ma_periods[0]}"
        long_ma = f"{ma_type}{ma_periods[1]}"
        
        # Create signal when short MA crosses above long MA (buy=1)
        df_clean['signal'] = 0
        df_clean.loc[df_clean[short_ma] > df_clean[long_ma], 'signal'] = 1
        df_clean.loc[df_clean[short_ma] < df_clean[long_ma], 'signal'] = -1
        
        # Create entries and exits
        df_clean['position'] = df_clean['signal']
        # Get position changes only
        df_clean['position_change'] = df_clean['position'].diff()
        
        # Generate vectorbt entries and exits signals
        entries = df_clean['position_change'] == 2  # From -1 to 1 (crossing from sell to buy)
        entries = entries | ((df_clean['position_change'] == 1) & (df_clean['position'].shift(1) == 0))  # From 0 to 1
        
        exits = df_clean['position_change'] == -2  # From 1 to -1 (crossing from buy to sell)
        exits = exits | ((df_clean['position_change'] == -1) & (df_clean['position'].shift(1) == 0))  # From 0 to -1
        
        # Convert boolean entries to entries in case they're not
        entries = entries.astype(bool)
        exits = exits.astype(bool)
        
        # Ensure we don't have entries and exits on the same bar
        prev_entries = entries.shift(1).fillna(False) # Assuming we start not entered (False)
        entries = entries & ~prev_entries
        
        # Run the backtest
        with start_action(action_type="backtest", phase="portfolio_execution") as backtest_action:
            portfolio = vbt.Portfolio.from_signals(
                df_clean['close'],
                entries=entries,
                exits=exits,
                init_cash=10000,
                fees=0.001,  # 0.1% trading fee
            )
            
            # Apply trailing stop-loss if specified
            if stop_loss_pct is not None and stop_loss_pct > 0:
                backtest_action.log(message_type="stop_loss", message=f"Applying {stop_loss_pct}% trailing stop-loss.")
                
                # Calculate the trailing stop price series
                trail_points = stop_loss_pct / 100
                portfolio = vbt.Portfolio.from_signals(
                    df_clean['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=10000,
                    fees=0.001,  # 0.1% trading fee
                    sl_trail=trail_points,  # Trailing stop loss specified as fraction
                )
        
        # Count signals
        buy_signals = (df_clean['signal'] == 1).sum()
        sell_signals = (df_clean['signal'] == -1).sum()
        hold_signals = (df_clean['signal'] == 0).sum()
        action.log(message_type="signal_distribution", message=f"Signal distribution: Buy={buy_signals}, Sell={sell_signals}, Hold={hold_signals}")
        
        # Extract trades
        if hasattr(portfolio, 'trades'):
            trades = portfolio.trades.records
            with start_action(action_type="backtest", phase="portfolio_trades") as trades_action:
                trades_action.log(message_type="trades_summary", message="Portfolio Trades")
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    
                    # Convert trade size to position size
                    if 'size' in trades_df.columns:
                        # Calculate average entry and exit prices
                        trades_df['entry_value'] = trades_df['size'] * trades_df['entry_price']
                        trades_df['exit_value'] = trades_df['size'] * trades_df['exit_price']
                        trades_df['avg_entry_price'] = trades_df['entry_value'] / trades_df['size']
                        trades_df['avg_exit_price'] = trades_df['exit_value'] / trades_df['size']
                    else:
                        trades_df['avg_entry_price'] = trades_df['entry_price']
                        trades_df['avg_exit_price'] = trades_df['exit_price']
                    
                    # Convert index values to timestamps
                    if 'entry_idx' in trades_df.columns and df_clean is not None:
                        trades_df['entry_time'] = df_clean.index[trades_df['entry_idx']]
                        trades_df['exit_time'] = df_clean.index[trades_df['exit_idx']]
                        
                    # Format the DataFrame for display  
                    display_cols = ['avg_entry_price', 'avg_exit_price', 'pnl', 'return']
                    if 'entry_time' in trades_df.columns:
                        display_cols = ['entry_time', 'exit_time'] + display_cols
                        
                    display_trades = trades_df[display_cols]
                    display_trades = display_trades.rename(columns={
                        'entry_time': 'Entry Timestamp',
                        'exit_time': 'Exit Timestamp',
                        'avg_entry_price': 'Avg Entry Price',
                        'avg_exit_price': 'Avg Exit Price',
                        'pnl': 'PnL',
                        'return': 'Return'
                    })
                    display_trades['Status'] = 'Closed'
                    
                    # Add a 'Status' column to indicate if the trade is open or closed
                    # But check first if needed columns exist in trades_df
                    if 'status' in trades_df.columns:
                        display_trades['Status'] = trades_df['status'].apply(lambda x: 'Open' if x == 0 else 'Closed')
                        
                    trades_action.log(message_type="trades_details", trades=str(display_trades))
                else:
                    trades_action.log(message_type="trades_empty", message="No trades executed.")
        
        # Get portfolio stats
        stats = portfolio.stats()
        
        # Show available stats keys for debugging
        with start_action(action_type="backtest", phase="portfolio_stats") as stats_action:
            stats_action.log(message_type="stats_keys", available_stats=str(list(stats.index)))
            
            # Calculate return manually for verification
            try:
                total_return = (portfolio.final_value() - portfolio.init_cash) / portfolio.init_cash
                stats_action.log(message_type="return_verification", message=f"Manual Return Calc: (Final: {portfolio.final_value():.2f} - Init: {portfolio.init_cash}) / Init: {portfolio.init_cash} = {total_return:.4f}")
            except Exception as e:
                stats_action.log(message_type="return_error", message=f"Error calculating manual return: {e}")
                total_return = 0.0
        
        # Get relevant metrics
        total_return_pct = stats["Total Return [%]"] if "Total Return [%]" in stats else 0
        sharpe_ratio = stats["Sharpe Ratio"] if "Sharpe Ratio" in stats else 0
        max_drawdown = stats["Max Drawdown [%]"] if "Max Drawdown [%]" in stats else 0
        win_rate = stats["Win Rate [%]"] if "Win Rate [%]" in stats else 0
        
        # Get trades count
        trades_count = len(portfolio.trades.records) if hasattr(portfolio, 'trades') else 0
        
        # Prepare close prices and equity curve for plotting
        updated_dates = df_clean.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        updated_close = df_clean['close'].tolist()
        signals = df_clean['signal'].tolist()
        equity_curve = portfolio.value().tolist()
        
        # Create MA values for the response
        ma_values = {
            'short': {
                'period': ma_periods[0],
                'type': ma_type,
                'values': df_clean[f'{ma_type}{ma_periods[0]}'].tolist()
            },
            'long': {
                'period': ma_periods[1],
                'type': ma_type,
                'values': df_clean[f'{ma_type}{ma_periods[1]}'].tolist()
            }
        }
        
        return MACrossoverResponse(
            symbol=symbol,
            dates=updated_dates,
            close_prices=updated_close,
            signals=signals,
            equity_curve=equity_curve,
            ma_values=ma_values,
            stats=StrategyStats(
                total_return_pct=float(total_return_pct),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown_pct=float(max_drawdown),
                win_rate_pct=float(win_rate),
                trades_count=trades_count
            )
        )


@mcp.tool()
async def generate_ma_chart(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 180,
    fast_ma: int = 20,
    slow_ma: int = 50,
    ma_type: str = "EMA"
) -> ImageContent:
    """
    Generate a chart showing price, moving averages and signals for the specified token.
    Returns the chart as an image.
    
    Args:
        symbol: Token symbol (e.g., "BTC")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        fast_ma: Period for the fast moving average
        slow_ma: Period for the slow moving average
        ma_type: Type of moving average ("EMA" or "SMA")
        
    Returns:
        ImageContent object containing the chart as base64-encoded image
    """
    # First, run the strategy to get the data
    with start_action(action_type="generate_ma_chart", symbol=symbol, 
                          ma_type=ma_type, fast_ma=fast_ma, slow_ma=slow_ma) as action:
        result = await ma_crossover_strategy(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            ma_type=ma_type,
            ma_periods=(fast_ma, slow_ma),
            stop_loss_pct=10.0,  # 10% trailing stop loss
        )
        
        # Log data details
        action.log(message_type="data_received", message="Data received by generate_ma_chart")
        action.log(message_type="data_stats", data_points=str(len(result.dates)))
        action.log(message_type="data_sample", first_5_dates=str(result.dates[:5]))
        action.log(message_type="data_sample", first_5_prices=str(result.close_prices[:5]))
        action.log(message_type="data_sample", first_5_signals=str(result.signals[:5]))
        action.log(message_type="data_sample", first_5_equity=str(result.equity_curve[:5]))
        action.log(message_type="data_sample", last_5_dates=str(result.dates[-5:]))
        action.log(message_type="data_sample", last_5_prices=str(result.close_prices[-5:]))
        action.log(message_type="data_sample", last_5_signals=str(result.signals[-5:]))
        action.log(message_type="data_sample", last_5_equity=str(result.equity_curve[-5:]))
        
        # Create a DataFrame from the result data for plotting
        plot_data = {
            'date': pd.to_datetime(result.dates),
            'close': result.close_prices,
            'signals': result.signals,
            'equity': result.equity_curve,
            'short_ma': result.ma_values['short']['values'],
            'long_ma': result.ma_values['long']['values']
        }
        df_plot = pd.DataFrame(plot_data)
        df_plot.set_index('date', inplace=True)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # --- Top Subplot (Price, MAs, Signals) --- 
        ax1.plot(df_plot.index, df_plot['close'], label='Price', color='blue', linewidth=1) # Thinner price line
        
        # Plot the MAs directly from the DataFrame
        short_ma_name = f"{ma_type}{fast_ma}"
        long_ma_name = f"{ma_type}{slow_ma}"
        ax1.plot(df_plot.index, df_plot['short_ma'], label=short_ma_name, color='orange', linewidth=1.5)
        ax1.plot(df_plot.index, df_plot['long_ma'], label=long_ma_name, color='purple', linewidth=1.5)
        
        # Plot buy signals (where signal == 1)
        buy_points = df_plot[df_plot['signals'] == 1]
        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals (where signal == -1) -> Use 'close' price for y-coordinate
        sell_points = df_plot[df_plot['signals'] == -1]
        if not sell_points.empty:
            ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        # Set labels and title for top subplot
        ax1.set_title(f"{symbol} Price with {ma_type} Crossover Signals ({fast_ma}/{slow_ma})")
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)
        
        # --- Bottom Subplot (Equity Curve) --- 
        ax2.plot(df_plot.index, df_plot['equity'], color='green', label='Portfolio Value')
        # Add initial capital line
        ax2.axhline(y=10000, color='gray', linestyle='--', label='Initial Capital') 
        
        # Set labels for bottom subplot
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Instead of saving to file, save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Close the plot to free memory
        plt.close(fig)
        
        action.log(message_type="chart_generated", message="Chart image generated and encoded successfully")
        
        # Return as ImageContent
        return ImageContent(
            type="image",
            data=img_str,
            mimeType="image/png"
        )


@mcp.tool()
async def generate_ma_chart_as_file(
    symbol: str,
    output_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 180,
    fast_ma: int = 20,
    slow_ma: int = 50,
    ma_type: str = "EMA"
) -> str:
    """
    Generate a chart showing price, moving averages and signals for the specified token.
    Saves the chart as an image file at the specified path.
    
    Args:
        symbol: Token symbol (e.g., "BTC")
        output_path: File path to save the image (if None, uses default directory)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        fast_ma: Period for the fast moving average
        slow_ma: Period for the slow moving average
        ma_type: Type of moving average ("EMA" or "SMA")
        
    Returns:
        Path where the image was saved
    """
    from datetime import datetime
    
    with start_action(action_type="generate_ma_chart_as_file", symbol=symbol, 
                    ma_type=ma_type, fast_ma=fast_ma, slow_ma=slow_ma) as action:
        # First, run the strategy to get the data
        action.log(message_type="strategy_start", message=f"Running strategy for {symbol} with {ma_type}{fast_ma}/{slow_ma}")
        result = await ma_crossover_strategy(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            ma_type=ma_type,
            ma_periods=(fast_ma, slow_ma),
            stop_loss_pct=10.0,  # 10% trailing stop loss
        )
        
        # Create a DataFrame from the result data for plotting
        with start_action(action_type="chart_creation", phase="data_preparation") as chart_action:
            plot_data = {
                'date': pd.to_datetime(result.dates),
                'close': result.close_prices,
                'signals': result.signals,
                'equity': result.equity_curve,
                'short_ma': result.ma_values['short']['values'],
                'long_ma': result.ma_values['long']['values']
            }
            df_plot = pd.DataFrame(plot_data)
            df_plot.set_index('date', inplace=True)
            chart_action.log(message_type="data_prep", message="Created plot dataframe", data_points=str(len(df_plot)))
        
        # Create figure with 2 subplots
        with start_action(action_type="chart_creation", phase="plotting") as plot_action:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # --- Top Subplot (Price, MAs, Signals) --- 
            ax1.plot(df_plot.index, df_plot['close'], label='Price', color='blue', linewidth=1)
            
            # Plot the MAs directly from the DataFrame
            short_ma_name = f"{ma_type}{fast_ma}"
            long_ma_name = f"{ma_type}{slow_ma}"
            ax1.plot(df_plot.index, df_plot['short_ma'], label=short_ma_name, color='orange', linewidth=1.5)
            ax1.plot(df_plot.index, df_plot['long_ma'], label=long_ma_name, color='purple', linewidth=1.5)
            
            # Plot buy signals (where signal == 1)
            buy_points = df_plot[df_plot['signals'] == 1]
            if not buy_points.empty:
                ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', s=100, label='Buy Signal', zorder=5)
                plot_action.log(message_type="signal_points", message=f"Added {len(buy_points)} buy signals to chart")
            
            # Plot sell signals (where signal == -1)
            sell_points = df_plot[df_plot['signals'] == -1]
            if not sell_points.empty:
                ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', s=100, label='Sell Signal', zorder=5)
                plot_action.log(message_type="signal_points", message=f"Added {len(sell_points)} sell signals to chart")
            
            # Set labels and title for top subplot
            ax1.set_title(f"{symbol} Price with {ma_type} Crossover Signals ({fast_ma}/{slow_ma})")
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True)
            
            # --- Bottom Subplot (Equity Curve) --- 
            ax2.plot(df_plot.index, df_plot['equity'], color='green', label='Portfolio Value')
            ax2.axhline(y=10000, color='gray', linestyle='--', label='Initial Capital')
            
            # Set labels for bottom subplot
            ax2.set_title('Equity Curve')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value ($)')
            ax2.legend()
            ax2.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            plot_action.log(message_type="plot_complete", message="Completed chart creation")
        
        # Determine the output path
        with start_action(action_type="chart_creation", phase="file_saving") as save_action:
            if output_path is None:
                # Navigate to project root (assuming we're in src/tokens_mcp)
                project_root = Path(__file__).parent.parent.parent.absolute() #in the future we will use the config file get image path
                
                # Create images directory in project root if it doesn't exist
                images_dir = project_root / "images"
                images_dir.mkdir(exist_ok=True, parents=True)
                save_action.log(message_type="directory_creation", message=f"Created images directory at {images_dir}")
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_{ma_type}{fast_ma}_{ma_type}{slow_ma}_{timestamp}.png"
                output_path = str(images_dir / filename)
                save_action.log(message_type="filename_generation", message=f"Generated filename: {filename}")
            else:
                # If custom path is provided, ensure its directory exists
                output_path_obj = Path(output_path)
                output_dir = output_path_obj.parent
                if output_dir != Path('.'):  # Only if there's a directory component
                    output_dir.mkdir(exist_ok=True, parents=True)
                    save_action.log(message_type="custom_directory", message=f"Using custom output directory: {output_dir}")
            
            # Save the figure to the file
            plt.savefig(output_path, format='png', dpi=300)
            save_action.log(message_type="file_saved", message=f"Saved chart to {output_path}")
            
            # Close the plot to free memory
            plt.close(fig)
        
        action.log(message_type="process_complete", message=f"Chart generation complete: {output_path}")
        return output_path


async def test_api_functionality():
    """
    Test function to verify the Token Metrics API functionality.
    Replicates calls from the examples to verify the API key is working.
    """
    with start_action(action_type="test_api_functionality") as action:
        action.log(message_type="test_start", message="TESTING TOKEN METRICS API FUNCTIONALITY")
        
        # Check if API key is set
        api_key = os.getenv("TOKEN_METRICS_API_KEY")
        if not api_key:
            action.log(message_type="api_key_missing", message="TOKEN_METRICS_API_KEY not set in environment")
            action.log(message_type="api_key_missing", message="Please set your Token Metrics API key in .env file")
            return
        else:
            action.log(message_type="api_key_found", message=f"API Key found: {api_key[:5]}...{api_key[-5:]}")
        
        # Create client instance
        client = TokenMetricsClient(api_key=api_key)
        
        try:
            # 1. Test tokens endpoint (get info for BTC and ETH)
            with start_action(action_type="api_test", phase="tokens_info") as tokens_action:
                tokens_action.log(message_type="test_phase", message="TESTING TOKENS INFO (BTC and ETH)")
                # Get information for Bitcoin and Ethereum
                tokens = client.tokens.get(symbol="BTC,ETH")
                tokens_count = len(tokens.get('data', []))
                tokens_action.log(message_type="tokens_count", message=f"Found {tokens_count} tokens")

                # Convert to DataFrame for easier exploration
                tokens_df = client.tokens.get_dataframe(symbol="BTC,ETH")
                tokens_action.log(message_type="tokens_data", message=f"DataFrame head: {tokens_df.head().to_string()}")

                tokens_action.log(message_type="tokens_success", message=f"Successfully retrieved info for {tokens_df.shape[0]} tokens")
                
                # Show a sample of the data
                if tokens_df.shape[0] > 0:
                    sample_token = tokens_df.iloc[0]
                    tokens_action.log(message_type="sample_token", message=f"Sample token: {sample_token.get('TOKEN_SYMBOL', 'Unknown')}")
                    tokens_action.log(message_type="sample_token_detail", message=f"Name: {sample_token.get('TOKEN_NAME', 'Unknown')}")
                    tokens_action.log(message_type="sample_token_detail", message=f"ID: {sample_token.get('TOKEN_ID', 'Unknown')}")
            
            # 2. Test daily OHLCV endpoint
            with start_action(action_type="api_test", phase="ohlcv_data") as ohlcv_action:
                ohlcv_action.log(message_type="test_phase", message="TESTING DAILY OHLCV (BTC for last 30 days)")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
                ohlcv_action.log(message_type="fetch_request", message=f"Fetching OHLCV data for BTC from {start_date} to {end_date}")

                # Get OHLCV data - the SDK will automatically handle the 29-day limit and show a progress bar
                ohlcv_df = client.hourly_ohlcv.get_dataframe(
                    symbol="BTC", 
                    startDate=start_date,
                    endDate=end_date
                )
                ohlcv_action.log(message_type="ohlcv_data", message=f"DataFrame head: {ohlcv_df.head().to_string()}")
                
                ohlcv_result = await get_daily_ohlcv_data("BTC", start_date=start_date, end_date=end_date)
                ohlcv_count = len(ohlcv_result.get("data", []))
                ohlcv_action.log(message_type="data_stats", message=f"Retrieved {ohlcv_count} OHLCV data points")
                ohlcv_action.log(message_type="ohlcv_success", message=f"Successfully retrieved {ohlcv_count} OHLCV data points")
                
                # Show a sample of the OHLCV data
                if ohlcv_count > 0:
                    sample_ohlcv = ohlcv_result.get("data", [])[0]
                    ohlcv_action.log(message_type="sample_ohlcv", message=f"Sample OHLCV data for {sample_ohlcv.get('TOKEN_SYMBOL', 'Unknown')}:")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"Date: {sample_ohlcv.get('DATE', 'Unknown')}")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"Open: {sample_ohlcv.get('OPEN', 'Unknown')}")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"High: {sample_ohlcv.get('HIGH', 'Unknown')}")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"Low: {sample_ohlcv.get('LOW', 'Unknown')}")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"Close: {sample_ohlcv.get('CLOSE', 'Unknown')}")
                    ohlcv_action.log(message_type="sample_ohlcv_detail", message=f"Volume: {sample_ohlcv.get('VOLUME', 'Unknown')}")
            
            # 3. Test trading signals
            with start_action(action_type="api_test", phase="trading_signals") as signals_action:
                signals_action.log(message_type="test_phase", message="TESTING TRADING SIGNALS (BTC for last 30 days)")
                signals_result = await get_trading_signals("BTC", start_date=start_date, end_date=end_date)
                signals_count = len(signals_result.get("data", []))
                signals_action.log(message_type="data_stats", message=f"Retrieved {signals_count} trading signals")
                signals_action.log(message_type="signals_success", message=f"Successfully retrieved {signals_count} trading signals")
                
                # Show a sample of the trading signals
                if signals_count > 0:
                    sample_signal = signals_result.get("data", [])[0]
                    signals_action.log(message_type="sample_signal", message=f"Sample signal: {sample_signal.get('SIGNAL', 'Unknown')}")
                    signals_action.log(message_type="sample_signal_detail", message=f"Token: {sample_signal.get('TOKEN_SYMBOL', 'Unknown')}")
                    signals_action.log(message_type="sample_signal_detail", message=f"Date: {sample_signal.get('DATE', 'Unknown')}")
                    signals_action.log(message_type="sample_signal_detail", message=f"Signal: {sample_signal.get('SIGNAL', 'Unknown')}")
                    signals_action.log(message_type="sample_signal_detail", message=f"Source: {sample_signal.get('SOURCE', 'Unknown')}")
            
            action.log(message_type="test_complete", message="API TEST COMPLETE: All endpoints working correctly")
            
        except Exception as e:
            action.log(message_type="test_error", message=f"ERROR: Failed to test API: {str(e)}", error=str(e))
            
        action.log(message_type="test_end", message="Test functionality complete")