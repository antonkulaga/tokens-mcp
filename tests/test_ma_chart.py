#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import base64  # Import base64

# Import the generate_ma_chart function directly from server.py
from server import generate_ma_chart

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
    print(f"Generating {ma_type} chart ({fast_ma}/{slow_ma}) for {symbol}...")
    
    # Set default dates if not provided
    today = datetime.now()
    if not end_date:
        end_date = today.strftime("%Y-%m-%d")
    if not start_date:
        start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Call the function directly and get the ImageContent result
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
    
    print(f"Chart saved to {output_filename}")

async def main():
    """
    Main function to run the chart generator test
    """
    try:
        # Test with ETC and EMA 20/50
        await test_ma_chart(symbol="ETC", ma_type="EMA", fast_ma=20, slow_ma=50)
        
        # Test with ETC and SMA 20/50
        await test_ma_chart(symbol="ETC", ma_type="SMA", fast_ma=20, slow_ma=50)
        
        # Test with ETC and EMA 10/30
        await test_ma_chart(symbol="ETC", ma_type="EMA", fast_ma=10, slow_ma=30)
        
        print("All charts generated successfully!")
        
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 