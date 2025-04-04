#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np

# Import the MA crossover strategy function directly from server.py
from server import ma_crossover_strategy

async def test_strategy(symbol="BTC", days_back=60, fast_ma=20, slow_ma=50, ma_type="EMA"):
    """
    Run the MA crossover strategy directly
    
    Args:
        symbol: Token symbol
        days_back: Number of days to look back
        fast_ma: Period for fast MA
        slow_ma: Period for slow MA
        ma_type: Type of MA (EMA or SMA)
        
    Returns:
        Strategy results
    """
    print(f"Running {ma_type} crossover strategy ({fast_ma}/{slow_ma}) for {symbol} over past {days_back} days...")
    
    # Call the function directly
    result = await ma_crossover_strategy(
        symbol=symbol,
        timeframe="1D",
        days_back=days_back,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        ma_type=ma_type
    )
    
    # Convert to dict for easier handling - use model_dump() for Pydantic v2
    result_dict = result.model_dump()
    return result_dict

def print_strategy_stats(result):
    """
    Print the strategy statistics
    
    Args:
        result: Strategy result
    """
    # Print strategy information
    print("\n" + "="*60)
    print(f"STRATEGY: {result['ma_type']} Crossover ({result['fast_ma']}/{result['slow_ma']})")
    print(f"SYMBOL: {result['symbol']} (Timeframe: {result['timeframe']})")
    print(f"PERIOD: {result['start_date']} to {result['end_date']} ({len(result['dates'])} days)")
    print("="*60)
    
    # Check data
    if not result['dates']:
        print("ERROR: No date data returned!")
        return
        
    # Print first and last few dates to check range
    dates = pd.to_datetime(result['dates'])
    print(f"First few dates: {dates[:5].tolist()}")
    print(f"Last few dates: {dates[-5:].tolist()}")
    
    # Examine signal distribution
    signals = np.array(result['signals'])
    buy_signals = sum(signals == 1)
    sell_signals = sum(signals == -1)
    hold_signals = sum(signals == 0)
    print(f"Signal distribution: Buy={buy_signals}, Sell={sell_signals}, Hold={hold_signals}")
    
    # Debugging: print the entire stats structure
    print("\nStats structure:")
    stats = result['stats']
    print(json.dumps(stats, indent=2))
    
    # Print performance metrics with error handling
    try:
        print(f"\nTotal Return: {stats.get('total_return', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {stats.get('max_drawdown', 0)*100:.2f}%")
        print(f"Win Rate: {stats.get('win_rate', 0)*100:.2f}%")
        print(f"Total Trades: {stats.get('total_trades', 0)}")
        
        if stats.get('profit_factor'):
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
        if stats.get('expectancy'):
            print(f"Expectancy: ${stats['expectancy']:.2f}")
    except Exception as e:
        print(f"Error displaying stats: {e}")
        
    print("="*60)

def plot_strategy_results(result):
    """
    Plot the strategy results
    
    Args:
        result: Strategy result
    """
    try:
        # Check if we have data
        if not result['dates'] or not result['close_prices']:
            print("No data to plot!")
            return
            
        # Create a DataFrame with the data
        df = pd.DataFrame({
            'close': result['close_prices'],
            'signals': result['signals'],
            'equity': result['equity_curve']
        }, index=pd.to_datetime(result['dates']))
        
        # Print DataFrame info
        print("\nData summary:")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Number of data points: {len(df)}")
        print(f"Price range: {min(df['close']):.2f} to {max(df['close']):.2f}")
        print(f"Equity range: {min(df['equity']):.2f} to {max(df['equity']):.2f}")
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and signals in top subplot
        ax1.plot(df.index, df['close'], label='Price', color='blue')
        
        # Plot buy signals
        buy_points = df[df['signals'] == 1]
        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_points = df[df['signals'] == -1]
        if not sell_points.empty:
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
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f"{result['symbol']}_{result['ma_type']}_{result['fast_ma']}_{result['slow_ma']}_strategy.png")
        print(f"Plot saved to {result['symbol']}_{result['ma_type']}_{result['fast_ma']}_{result['slow_ma']}_strategy.png")
        
        # Don't try to show in non-interactive environments
        if plt.isinteractive():
            plt.show()
            
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """
    Main function to run the strategy and display results
    """
    # Run strategy for BTC with default parameters
    try:
        # Test with more recent past data
        result = await test_strategy(symbol="BTC", days_back=365)
        
        # Debug: Print all keys in the result
        print("Result keys:", list(result.keys()))
        
        # Print stats
        print_strategy_stats(result)
        
        # Save results to file
        with open(f"BTC_strategy_results.json", "w") as f:
            json.dump(result, f, indent=2)
            print(f"Results saved to BTC_strategy_results.json")
        
        # Plot results
        plot_strategy_results(result)
        
        print("\nTesting strategy with different parameters...")
        
        # Test with SMA instead of EMA
        result_sma = await test_strategy(symbol="BTC", days_back=365, ma_type="SMA")
        print_strategy_stats(result_sma)
        
        # Test with different MA periods
        result_fast = await test_strategy(symbol="BTC", days_back=365, fast_ma=10, slow_ma=30)
        print_strategy_stats(result_fast)
        
    except Exception as e:
        print(f"Error testing strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 