#!/usr/bin/env python3
"""
Test module to demonstrate using the generate_ma_chart_as_file function
from the Token Metrics API server.
"""

import asyncio
import os
from dotenv import load_dotenv
from server import generate_ma_chart_as_file

# Load environment variables (needed for Token Metrics API key)
load_dotenv(override=True)

async def test_chart_generation():
    """
    Test function to demonstrate generating and saving charts as files.
    """
    print("Starting chart generation test...")
    
    # Check if API key is set
    api_key = os.getenv("TOKEN_METRICS_API_KEY")
    if not api_key:
        print("⚠️ ERROR: TOKEN_METRICS_API_KEY not set in environment")
        print("Please set your Token Metrics API key in .env file")
        return
    
    # Test cases for different cryptocurrencies and parameters
    test_cases = [
        # Default parameters for Bitcoin
        {"symbol": "BTC"},
        
        # Ethereum with custom MA periods
        {
            "symbol": "ETH", 
            "fast_ma": 10, 
            "slow_ma": 30,
            "days_back": 365  # 1 year of data
        },
        
        # Solana with SMA instead of EMA
        {
            "symbol": "SOL", 
            "ma_type": "SMA", 
            "fast_ma": 15, 
            "slow_ma": 40
        },
        
        # Custom output path in current directory
        {
            "symbol": "BNB",
            "output_path": "bnb_chart.png"
        }
    ]
    
    # Run each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case #{i}: {test_case}")
        try:
            # Generate and save the chart
            output_path = await generate_ma_chart_as_file(**test_case)
            print(f"✅ Chart saved successfully to: {output_path}")
        except Exception as e:
            print(f"❌ Error generating chart: {e}")
    
    print("\nChart generation test complete!")

if __name__ == "__main__":
    # Run the test function
    asyncio.run(test_chart_generation()) 