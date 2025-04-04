import asyncio
from server import (
    get_tokens_list, 
    get_token_data, 
    get_categories_list, 
    get_exchanges_list,
    get_cache_status
)

async def test_cache():
    """Test the caching functionality"""
    print("==== CACHE STATUS ====")
    status = get_cache_status()
    for cache_name, info in status.items():
        exists = "✅" if info["exists"] else "❌"
        valid = "✅" if info["valid"] else "❌"
        print(f"{cache_name}: Exists: {exists} | Valid: {valid}")
    
    print("\n==== LOADING FROM CACHE ====")
    # Test loading tokens from cache
    start_time = asyncio.get_event_loop().time()
    tokens = await get_tokens_list(use_cache=True)
    end_time = asyncio.get_event_loop().time()
    print(f"Loaded {len(tokens)} tokens from cache in {end_time - start_time:.4f} seconds")
    
    # Test loading Bitcoin data from cache
    btc = await get_token_data("BTC", use_cache=True)
    if btc:
        print(f"Found Bitcoin in cache: {btc.TOKEN_NAME}")
    else:
        print("Bitcoin not found in cache")
    
    # Test loading Ethereum data from cache
    eth = await get_token_data("ETH", use_cache=True)
    if eth:
        print(f"Found Ethereum in cache: {eth.TOKEN_NAME}")
    else:
        print("Ethereum not found in cache")
    
    # Test loading categories from cache
    categories = await get_categories_list(use_cache=True)
    print(f"Loaded {len(categories)} categories from cache")
    
    # Test loading exchanges from cache
    exchanges = await get_exchanges_list(use_cache=True)
    print(f"Loaded {len(exchanges)} exchanges from cache")

if __name__ == "__main__":
    asyncio.run(test_cache()) 