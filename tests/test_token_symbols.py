import asyncio
import os
from dotenv import load_dotenv
from tokens_mcp.helpers import fetch_tokens_list

async def find_popular_tokens():
    """
    Test if we can find popular tokens like BTC and ETH in the token list
    """
    print("Testing if we can find popular tokens...")
    
    # Popular token symbols to look for
    popular_tokens = ["BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX"]
    
    # Fetch tokens with a large limit
    print("Fetching tokens list...")
    tokens = await fetch_tokens_list(limit=10000)
    print(f"Fetched {len(tokens)} tokens")
    
    # Create a mapping of symbol to token
    token_map = {}
    invalid_symbols = 0
    
    for token in tokens:
        symbol = token.get("TOKEN_SYMBOL")
        if symbol and isinstance(symbol, str):  # Make sure symbol is a string
            # Store in both upper and original case for flexibility
            token_map[symbol.upper()] = token
            if symbol not in token_map:
                token_map[symbol] = token
        elif symbol is not None:
            invalid_symbols += 1
    
    if invalid_symbols > 0:
        print(f"Warning: Found {invalid_symbols} tokens with non-string symbols")
    
    print(f"Created mapping with {len(token_map)} unique token symbols")
    
    # Check for each popular token
    found_tokens = []
    missing_tokens = []
    
    for symbol in popular_tokens:
        if symbol in token_map:
            token = token_map[symbol]
            found_tokens.append({
                "symbol": symbol,
                "name": token.get("TOKEN_NAME"),
                "id": token.get("TOKEN_ID"),
                "type": token.get("TOKEN_TYPE")
            })
            print(f"✅ Found {symbol}: {token.get('TOKEN_NAME')} (ID: {token.get('TOKEN_ID')})")
        else:
            missing_tokens.append(symbol)
            print(f"❌ Could not find {symbol}")
    
    # Summary
    print(f"\nFound {len(found_tokens)} out of {len(popular_tokens)} popular tokens")
    if missing_tokens:
        print(f"Missing tokens: {', '.join(missing_tokens)}")
    
    # Check for alternative symbols for missing tokens
    if missing_tokens:
        print("\nSearching for alternative symbols...")
        
        alternatives = {
            "BTC": ["BITCOIN", "WBTC", "BTCB"],
            "ETH": ["ETHEREUM", "WETH", "BETH"],
            "USDT": ["TETHER"],
            "BNB": ["BINANCE", "WBNB"],
            "SOL": ["SOLANA", "WSOL"],
            "XRP": ["RIPPLE"],
            "ADA": ["CARDANO"],
            "DOGE": ["DOGECOIN"],
            "DOT": ["POLKADOT"],
            "AVAX": ["AVALANCHE", "WAVAX"]
        }
        
        for symbol in missing_tokens:
            if symbol in alternatives:
                found_alt = False
                for alt in alternatives[symbol]:
                    if alt in token_map:
                        print(f"✅ Found alternative for {symbol}: {alt} - {token_map[alt].get('TOKEN_NAME')}")
                        found_alt = True
                        break
                if not found_alt:
                    print(f"❌ No alternatives found for {symbol}")
    
    # Print total unique valid symbols
    valid_symbols = [t.get('TOKEN_SYMBOL', '').upper() for t in tokens 
                    if t.get('TOKEN_SYMBOL') and isinstance(t.get('TOKEN_SYMBOL'), str)]
    print(f"\nTotal unique valid token symbols: {len(set(valid_symbols))}")
    
    # Look for tokens that might be sorted at the end
    print("\nChecking tokens sorted alphabetically...")
    valid_tokens = [t for t in tokens if t.get('TOKEN_SYMBOL') and isinstance(t.get('TOKEN_SYMBOL'), str)]
    sorted_tokens = sorted(valid_tokens, key=lambda x: x.get("TOKEN_SYMBOL", "").upper())
    
    if len(sorted_tokens) >= 3:
        print(f"First tokens: {sorted_tokens[0].get('TOKEN_SYMBOL')} - {sorted_tokens[1].get('TOKEN_SYMBOL')} - {sorted_tokens[2].get('TOKEN_SYMBOL')}")
        print(f"Last tokens: {sorted_tokens[-3].get('TOKEN_SYMBOL')} - {sorted_tokens[-2].get('TOKEN_SYMBOL')} - {sorted_tokens[-1].get('TOKEN_SYMBOL')}")
    
    # Print a sample of tokens to debug data quality
    print("\nSample of 5 random tokens:")
    import random
    sample_tokens = random.sample(tokens, min(5, len(tokens)))
    for i, token in enumerate(sample_tokens):
        print(f"{i+1}. Symbol: {token.get('TOKEN_SYMBOL')} ({type(token.get('TOKEN_SYMBOL')).__name__})")
        print(f"   Name: {token.get('TOKEN_NAME')}")
        print(f"   ID: {token.get('TOKEN_ID')}")

if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(find_popular_tokens()) 