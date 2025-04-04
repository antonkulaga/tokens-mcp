import asyncio
import os
import json
from dotenv import load_dotenv
from helpers import fetch_from_api

async def test_pagination_limit():
    """
    Test if the limit parameter is correctly enforced in the Token Metrics API
    and if we can get different numbers of tokens with different limit values
    """
    print("Testing if limit parameter is correctly enforced in Token Metrics API")
    
    # Test different limit values
    limits = [5, 10, 50, 100, 1000]
    results = []  # Store results for later analysis
    
    for limit in limits:
        print(f"\nTesting with limit={limit}")
        response = await fetch_from_api("tokens", {"limit": limit})
        items = response.get("data", [])
        item_count = len(items)
        results.append((limit, item_count))
        
        print(f"Requested {limit} items, received {item_count} items")
        
        # Check if the returned number of items matches the requested limit
        if item_count == limit:
            print(f"✅ Limit {limit} correctly enforced")
        elif item_count < limit:
            print(f"⚠️ Received fewer items ({item_count}) than requested ({limit})")
            print("   This may be normal if we've reached the end of available data")
        else:
            print(f"❌ Received more items ({item_count}) than requested ({limit})")
    
    # Test with very large limit
    large_limit = 5000
    print(f"\nTesting with a very large limit: {large_limit}")
    large_response = await fetch_from_api("tokens", {"limit": large_limit})
    large_items = large_response.get("data", [])
    large_count = len(large_items)
    
    print(f"Requested {large_limit} items, received {large_count} items")
    
    # Check if API has a maximum limit cap
    if large_count < large_limit:
        print(f"⚠️ API may have a maximum limit cap around {large_count} items")
    
    # Test with no limit specified (default limit)
    print("\nTesting with no limit specified (default limit)")
    default_response = await fetch_from_api("tokens", {})
    default_items = default_response.get("data", [])
    default_count = len(default_items)
    
    print(f"No limit specified, received {default_count} items")
    print(f"Default limit appears to be {default_count}")
    
    # Check total count if available
    total_count = large_response.get("meta", {}).get("total", 0)
    if total_count:
        print(f"\nAPI reports a total of {total_count} items available")
        print(f"Maximum items we were able to fetch in one request: {large_count}")
        
        if large_count < total_count:
            pages_needed = (total_count + large_count - 1) // large_count
            print(f"To fetch all {total_count} items, approximately {pages_needed} requests would be needed")
    
    print("\nTest complete!")
    
    # Print conclusion
    print("\n====== CONCLUSION ======")
    # Calculate largest confirmed working limit from our results
    working_limits = [l for l, c in results if l == c]
    max_working_limit = max(working_limits) if working_limits else 0
    
    print(f"Limit parameter summary:")
    print(f"- Largest confirmed working limit: {max_working_limit}")
    print(f"- Largest tested limit: {large_limit} (returned {large_count} items)")
    print(f"- Default limit: {default_count}")
    
    if large_count < large_limit:
        print(f"The API appears to cap results at around {large_count} items per request")
        print("To fetch all items, you'll need to use pagination (multiple requests)")
    else:
        print("The API appears to honor large limit values, but pagination may still be required for very large datasets")

if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(test_pagination_limit()) 