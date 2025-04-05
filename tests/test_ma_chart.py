#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import base64  # Import base64
from eliot import start_action, to_file, Message
import os
from pathlib import Path

# Import the generate_ma_chart function directly from server.py
from tokens_mcp.server import generate_ma_chart

async def test_ma_chart(
    symbol: str = "BTC", 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    days_back: int = 180,
    fast_ma: int = 20,
    slow_ma: int = 50,
    ma_type: str = "EMA"
) -> None:
    """
    Test the MA chart generator tool
    
    Args:
        symbol: Token symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days_back: Number of days to look back if start_date not provided
        fast_ma: Period for fast MA
        slow_ma: Period for slow MA
        ma_type: Type of MA (EMA or SMA)
    """
    with start_action(action_type="test_ma_chart", 
                     symbol=symbol, 
                     ma_type=ma_type, 
                     fast_ma=fast_ma,
                     slow_ma=slow_ma) as action:
        
        # Set default dates if not provided
        today = datetime.now()
        if not end_date:
            end_date = today.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        action.add_success_fields(start_date=start_date, end_date=end_date)
        
        # Call the function directly and get the ImageContent result
        with start_action(action_type="generate_chart"):
            image_content = await generate_ma_chart(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                ma_type=ma_type
            )
            
            # Extract the base64 data
            b64_data = image_content.data
            
            # Decode the base64 data
            image_data = base64.b64decode(b64_data)
            
            # Save to file
            output_filename = f"{symbol}_{ma_type}_{fast_ma}_{slow_ma}_chart.png"
            with open(output_filename, "wb") as f:
                f.write(image_data)
            
            action.add_success_fields(output_filename=output_filename)

async def main():
    """
    Main function to run the chart generator test
    """
    with start_action(action_type="ma_chart_tests") as action:
        try:
            # Test with ETC and EMA 20/50
            with start_action(action_type="etc_ema_20_50"):
                await test_ma_chart(symbol="ETC", ma_type="EMA", fast_ma=20, slow_ma=50)
            
            # Test with ETC and SMA 20/50
            with start_action(action_type="etc_sma_20_50"):
                await test_ma_chart(symbol="ETC", ma_type="SMA", fast_ma=20, slow_ma=50)
            
            # Test with ETC and EMA 10/30
            with start_action(action_type="etc_ema_10_30"):
                await test_ma_chart(symbol="ETC", ma_type="EMA", fast_ma=10, slow_ma=30)
            
            action.add_success_fields(status="all_tests_completed")
            
        except Exception as e:
            action.add_exception(e)

if __name__ == "__main__":
    # Set up Eliot logging to file
    log_file = Path("ma_chart_test.log")
    to_file(open(log_file, "wb"))
    
    # Run the async main function
    asyncio.run(main()) 