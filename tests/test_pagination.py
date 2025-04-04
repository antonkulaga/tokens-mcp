import asyncio
import os
import json
from dotenv import load_dotenv
from helpers import fetch_from_api

async def test_token_pagination():
    """
    Test if pagination is working correctly by comparing page 0 and page 100
    with a high limit value (10000)
    """
    print("Testing token pagination with limit=10000")
    
    # Test page 0
    print("\nFetching page 0...")
    page0_response = await fetch_from_api("tokens", {"limit": 10000, "page": 0})
    page0_items = page0_response.get("data", [])
    page0_count = len(page0_items)
    
    print(f"Page 0 returned {page0_count} items")
    print(f"First item on page 0: {page0_items[0]['TOKEN_SYMBOL'] if page0_items else 'None'}")
    print(f"Last item on page 0: {page0_items[-1]['TOKEN_SYMBOL'] if page0_items else 'None'}")
    
    # Check if there's metadata about total items
    total_count = page0_response.get("meta", {}).get("total", 0)
    if total_count:
        print(f"API reports total of {total_count} items")
    
    # Test page 100 (should be empty or have different items if pagination works)
    print("\nFetching page 100...")
    page100_response = await fetch_from_api("tokens", {"limit": 10000, "page": 100})
    page100_items = page100_response.get("data", [])
    page100_count = len(page100_items)
    
    print(f"Page 100 returned {page100_count} items")
    print(f"First item on page 100: {page100_items[0]['TOKEN_SYMBOL'] if page100_items else 'None'}")
    print(f"Last item on page 100: {page100_items[-1]['TOKEN_SYMBOL'] if page100_items else 'None'}")
    
    # Check if pages are the same (which would indicate pagination is broken)
    if page0_count > 0 and page100_count > 0:
        # Compare first item from each page
        first_item_page0 = page0_items[0].get("TOKEN_ID") if page0_items else None
        first_item_page100 = page100_items[0].get("TOKEN_ID") if page100_items else None
        
        if first_item_page0 == first_item_page100:
            print("\n⚠️ PAGINATION ISSUE DETECTED: First items on page 0 and page 100 are identical")
            print("This suggests the API is ignoring the page parameter and always returning the same data")
        else:
            print("\n✅ PAGINATION SEEMS WORKING: Different data returned for different pages")
    elif page0_count > 0 and page100_count == 0:
        print("\n✅ PAGINATION SEEMS WORKING: Page 0 has data but page 100 is empty (expected if there are fewer than 1,000,000 items)")
    
    # Additional test - check two consecutive pages
    print("\nFetching page 1 for additional verification...")
    page1_response = await fetch_from_api("tokens", {"limit": 10000, "page": 1})
    page1_items = page1_response.get("data", [])
    page1_count = len(page1_items)
    
    print(f"Page 1 returned {page1_count} items")
    
    if page0_count > 0 and page1_count > 0:
        # Check if any items from page 0 also appear on page 1
        page0_ids = set(item.get("TOKEN_ID") for item in page0_items if item.get("TOKEN_ID"))
        page1_ids = set(item.get("TOKEN_ID") for item in page1_items if item.get("TOKEN_ID"))
        
        overlap = page0_ids.intersection(page1_ids)
        
        if overlap:
            print(f"⚠️ PAGINATION ISSUE: {len(overlap)} overlapping items found between page 0 and page 1")
        else:
            print("✅ No overlapping items between page 0 and page 1")
    
    print("\nTest complete!")

if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(test_token_pagination()) 