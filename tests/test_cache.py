import asyncio
import pytest
from eliot import start_action, to_file, Message
from tokens_mcp.server import (
    #get_tokens_list, #broken in API
    get_token_data, 
    get_categories_list, 
    get_exchanges_list,
    get_cache_status
)
from pycomfort.logging import to_nice_stdout

@pytest.mark.skip(reason="Method get_tokens_list in TokenMetrics API is temporarily broken")
async def test_cache():
    """Test the caching functionality"""
    with start_action(action_type="test_cache") as action:
        # Check cache status
        with start_action(action_type="check_cache_status"):
            status = get_cache_status()
            for cache_name, info in status.items():
                Message.log(
                    message_type="cache_info",
                    cache=cache_name,
                    exists=info["exists"],
                    valid=info["valid"]
                )
        
        # Test loading tokens from cache
        # TODO: uncomment when TokenMetrics guys will fix this method in their endpoint
        #with start_action(action_type="load_tokens"):
        #    start_time = asyncio.get_event_loop().time()
        #    tokens = await get_tokens_list(use_cache=True)
        #    end_time = asyncio.get_event_loop().time()
        #    action.add_success_fields(token_count=len(tokens), duration=end_time - start_time)
        
        # Test loading Bitcoin data from cache
        with start_action(action_type="load_token", symbol="BTC") as btc_action:
            btc = await get_token_data("BTC", use_cache=True)
            if btc:
                btc_action.add_success_fields(found=True, name=btc["TOKEN_NAME"])
            else:
                btc_action.add_success_fields(found=False)
        
        # Test loading Ethereum data from cache
        with start_action(action_type="load_token", symbol="ETH") as eth_action:
            eth = await get_token_data("ETH", use_cache=True)
            if eth:
                eth_action.add_success_fields(found=True, name=eth["TOKEN_NAME"])
            else:
                eth_action.add_success_fields(found=False)
        
        # Test loading categories from cache
        with start_action(action_type="load_categories") as categories_action:
            categories = await get_categories_list(use_cache=True)
            categories_action.add_success_fields(count=len(categories))
        
        # Test loading exchanges from cache
        with start_action(action_type="load_exchanges") as exchanges_action:
            exchanges = await get_exchanges_list(use_cache=True)
            exchanges_action.add_success_fields(count=len(exchanges))

if __name__ == "__main__":
    to_nice_stdout()
    asyncio.run(test_cache()) 