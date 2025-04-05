#!/usr/bin/env python3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from eliot import log_call, log_message, log_error, log_exception, log_warning, log_info

# Define the MCP API endpoint
MCP_ENDPOINT = "http://localhost:8000/api/tools/ma_crossover_strategy"

def run_ma_strategy(symbol="BTC", days_back=60, fast_ma=20, slow_ma=50, ma_type="EMA"):
    """
    Run the MA crossover strategy using the MCP tool
    
    Args:
        symbol: Token symbol
        days_back: Number of days to look back
        fast_ma: Period for fast MA
        slow_ma: Period for slow MA
        ma_type: Type of MA (EMA or SMA)
        
    Returns:
        Strategy results
    """
    # Prepare payload
    payload = {
        "symbol": symbol,
        "timeframe": "1D",
        "days_back": days_back,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "ma_type": ma_type
    }
    
    # Make request to MCP API
    print(f"Running {ma_type} crossover strategy ({fast_ma}/{slow_ma}) for {symbol} over past {days_back} days...")
    response = requests.post(MCP_ENDPOINT, json=payload)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    # Return JSON response
    return response.json()

def print_strategy_stats(result):
    """
    Print the strategy statistics
    
    Args:
        result: Strategy result from the MA crossover API
    """
    # Print strategy information
    print("\n" + "="*60)
    print(f"STRATEGY: {result['ma_type']} Crossover ({result['fast_ma']}/{result['slow_ma']})")
    print(f"SYMBOL: {result['symbol']} (Timeframe: {result['timeframe']})")
    print(f"PERIOD: {result['start_date']} to {result['end_date']} ({len(result['dates'])} days)")
    print("="*60)
    
    # Print performance metrics
    stats = result['stats']
    print(f"Total Return: {stats['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {stats['win_rate']*100:.2f}%")
    print(f"Total Trades: {stats['total_trades']}")
    
    if stats.get('profit_factor'):
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
    if stats.get('expectancy'):
        print(f"Expectancy: ${stats['expectancy']:.2f}")
    print("="*60)

def plot_strategy_results(result):
    """
    Plot the strategy results
    
    Args:
        result: Strategy result from the MA crossover API
    """
    # Create a DataFrame with the data
    df = pd.DataFrame({
        'close': result['close_prices'],
        'signals': result['signals'],
        'equity': result['equity_curve']
    }, index=pd.to_datetime(result['dates']))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and signals in top subplot
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    
    # Plot buy signals
    buy_points = df[df['signals'] == 1]
    ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_points = df[df['signals'] == -1]
    ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', s=100, label='Sell Signal')
    
    # Set labels and title for top subplot
    ax1.set_title(f"{result['symbol']} Price with {result['ma_type']} Crossover Signals ({result['fast_ma']}/{result['slow_ma']})")
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot equity curve in bottom subplot
    ax2.plot(df.index, df['equity'], color='green', label='Portfolio Value')
    ax2.axhline(y=10000, color='gray', linestyle='--', label='Initial Capital')
    
    # Set labels for bottom subplot
    ax2.set_title('Equity Curve')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and show plot
    plt.tight_layout()
    
    # Create images directory if it doesn't exist
    images_dir = Path(__file__).parents[1] / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Save the figure to the file
    filename = f"{result['symbol']}_{result['ma_type']}_{result['fast_ma']}_{result['slow_ma']}_strategy.png"
    output_path = images_dir / filename
    plt.savefig(output_path)
    print(f"Strategy plot saved to: {output_path}")
    
    plt.show()

def main():
    """
    Main function to run the strategy and display results
    """
    # Run strategy for BTC with default parameters
    result = run_ma_strategy(symbol="BTC", days_back=60)
    
    if result:
        # Print stats
        print_strategy_stats(result)
        
        # Plot results
        plot_strategy_results(result)
        
        # Save results to file
        images_dir = Path(__file__).parents[1] / "images"
        images_dir.mkdir(exist_ok=True)
        
        results_file = images_dir / f"{result['symbol']}_strategy_results.json"
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)
            print(f"Results saved to: {results_file}")
    
    print("\nTesting strategy with different parameters...")
    
    # Test with SMA instead of EMA
    result_sma = run_ma_strategy(symbol="BTC", days_back=60, ma_type="SMA")
    if result_sma:
        print_strategy_stats(result_sma)
    
    # Test with different MA periods
    result_fast = run_ma_strategy(symbol="BTC", days_back=60, fast_ma=10, slow_ma=30)
    if result_fast:
        print_strategy_stats(result_fast)

if __name__ == "__main__":
    main() 