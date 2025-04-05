#!/usr/bin/env python3
"""
Test module to demonstrate using the generate_ma_chart_as_file function
from the Token Metrics API server.
"""

import asyncio
import os
from dotenv import load_dotenv
from eliot import start_action
from tokens_mcp.server import generate_ma_chart_as_file
from pycomfort.logging import to_nice_stdout

# Load environment variables (needed for Token Metrics API key)
load_dotenv(override=True)

async def test_chart_generation():
    """
    Test function to demonstrate generating and saving charts as files.
    """
    with start_action(action_type="chart_generation_test") as action:
        # Check if API key is set
        api_key = os.getenv("TOKEN_METRICS_API_KEY")
        if not api_key:
            action.log(message_type="api_key_missing", description="TOKEN_METRICS_API_KEY not set in environment")
            action.add_success_fields(status="failed", reason="missing_api_key")
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
        
        success_count = 0
        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            with start_action(action_type="generate_chart", test_case_index=i, params=test_case) as test_action:
                try:
                    # Generate and save the chart
                    output_path = await generate_ma_chart_as_file(**test_case)
                    test_action.add_success_fields(output_path=output_path)
                    success_count += 1
                except Exception as e:
                    test_action.add_exception(e)
        
        # Add summary information
        action.add_success_fields(
            total_test_cases=len(test_cases),
            successful_tests=success_count,
            failed_tests=len(test_cases) - success_count
        )

if __name__ == "__main__":
    # Run the test function
    to_nice_stdout()
    asyncio.run(test_chart_generation()) 