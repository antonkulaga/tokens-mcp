#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from eliot import start_action, to_file, Message

# Import the MA crossover strategy function directly from server.py
from tokens_mcp.server import ma_crossover_strategy

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
    with start_action(action_type="test_strategy", 
                     symbol=symbol, 
                     days_back=days_back,
                     ma_type=ma_type,
                     fast_ma=fast_ma,
                     slow_ma=slow_ma) as action:
        
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
    with start_action(action_type="log_strategy_stats", 
                      symbol=result['symbol'], 
                      ma_type=result['ma_type'],
                      timeframe=result['timeframe']) as action:
        # Log strategy information
        Message.log(
            message_type="strategy_info",
            ma_type=result['ma_type'],
            fast_ma=result['fast_ma'],
            slow_ma=result['slow_ma'],
            symbol=result['symbol'],
            timeframe=result['timeframe'],
            start_date=result['start_date'],
            end_date=result['end_date'],
            days_count=len(result['dates'])
        )
        
        # Check data
        if not result['dates']:
            Message.log(message_type="error", description="No date data returned")
            action.add_success_fields(has_data=False)
            return
            
        # Log date range info
        dates = pd.to_datetime(result['dates'])
        action.add_success_fields(
            first_dates=dates[:5].strftime("%Y-%m-%d").tolist(),
            last_dates=dates[-5:].strftime("%Y-%m-%d").tolist()
        )
        
        # Log signal distribution
        signals = np.array(result['signals'])
        buy_signals = sum(signals == 1)
        sell_signals = sum(signals == -1)
        hold_signals = sum(signals == 0)
        
        Message.log(
            message_type="signal_distribution",
            buy_signals=int(buy_signals),
            sell_signals=int(sell_signals),
            hold_signals=int(hold_signals)
        )
        
        # Log performance metrics
        stats = result['stats']
        try:
            Message.log(
                message_type="performance_metrics",
                total_return=stats.get('total_return', 0) * 100,
                sharpe_ratio=stats.get('sharpe_ratio', 0),
                max_drawdown=stats.get('max_drawdown', 0) * 100,
                win_rate=stats.get('win_rate', 0) * 100,
                total_trades=stats.get('total_trades', 0),
                profit_factor=stats.get('profit_factor', None),
                expectancy=stats.get('expectancy', None)
            )
        except Exception as e:
            action.add_exception(e)

def plot_strategy_results(result):
    """
    Plot the strategy results
    
    Args:
        result: Strategy result
    """
    with start_action(action_type="plot_strategy", 
                     symbol=result['symbol'],
                     ma_type=result['ma_type']) as action:
        try:
            # Check if we have data
            if not result['dates'] or not result['close_prices']:
                Message.log(message_type="error", description="No data to plot")
                action.add_success_fields(has_data=False)
                return
                
            # Create a DataFrame with the data
            df = pd.DataFrame({
                'close': result['close_prices'],
                'signals': result['signals'],
                'equity': result['equity_curve']
            }, index=pd.to_datetime(result['dates']))
            
            # Log DataFrame info
            Message.log(
                message_type="data_summary",
                date_min=df.index.min().strftime("%Y-%m-%d"),
                date_max=df.index.max().strftime("%Y-%m-%d"),
                data_points=len(df),
                price_min=float(min(df['close'])),
                price_max=float(max(df['close'])),
                equity_min=float(min(df['equity'])),
                equity_max=float(max(df['equity']))
            )
            
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
            filename = f"{result['symbol']}_{result['ma_type']}_{result['fast_ma']}_{result['slow_ma']}_strategy.png"
            plt.savefig(filename)
            action.add_success_fields(plot_filename=filename)
                
        except Exception as e:
            action.add_exception(e)

async def main():
    """
    Main function to run the strategy and display results
    """
    with start_action(action_type="ma_strategy_tests") as action:
        try:
            # Test with more recent past data
            with start_action(action_type="btc_default_strategy") as btc_action:
                result = await test_strategy(symbol="BTC", days_back=365)
                
                # Log all keys in the result
                btc_action.add_success_fields(result_keys=list(result.keys()))
                
                # Print stats
                print_strategy_stats(result)
                
                # Save results to file
                results_file = f"BTC_strategy_results.json"
                with open(results_file, "w") as f:
                    json.dump(result, f, indent=2)
                    btc_action.add_success_fields(results_file=results_file)
                
                # Plot results
                plot_strategy_results(result)
            
            Message.log(message_type="testing_alternative_params")
            
            # Test with SMA instead of EMA
            with start_action(action_type="btc_sma_strategy"):
                result_sma = await test_strategy(symbol="BTC", days_back=365, ma_type="SMA")
                print_strategy_stats(result_sma)
            
            # Test with different MA periods
            with start_action(action_type="btc_fast_ma_strategy"):
                result_fast = await test_strategy(symbol="BTC", days_back=365, fast_ma=10, slow_ma=30)
                print_strategy_stats(result_fast)
            
        except Exception as e:
            action.add_exception(e)

if __name__ == "__main__":
    # Set up Eliot logging to file
    log_file = Path("ma_strategy_test.log")
    to_file(open(log_file, "wb"))
    
    # Run the async main function
    asyncio.run(main()) 