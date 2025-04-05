import asyncio
import os
import json
from dotenv import load_dotenv
from tokens_mcp.helpers import fetch_from_api

async def test_offset_pagination():
    """
    Test if the Token Metrics API uses offset-based pagination instead of page-based pagination
    """
    print("Testing if Token Metrics API uses offset-based pagination")
    
    limit = 10
    
    # Test with page parameter
    print("\nTest 1: Using page parameter")
    print("Fetching page 0 with limit 10...")
    page0_response = await fetch_from_api("tokens", {"limit": limit, "page": 0})
    page0_items = page0_response.get("data", [])
    
    print("Fetching page 1 with limit 10...")
    page1_response = await fetch_from_api("tokens", {"limit": limit, "page": 1})
    page1_items = page1_response.get("data", [])
    
    # Print first 3 tokens from each page for comparison
    print("\nFirst 3 tokens from page 0:")
    for i, item in enumerate(page0_items[:3]):
        print(f"{i+1}. {item.get('TOKEN_SYMBOL')} (ID: {item.get('TOKEN_ID')})")
    
    print("\nFirst 3 tokens from page 1:")
    for i, item in enumerate(page1_items[:3]):
        print(f"{i+1}. {item.get('TOKEN_SYMBOL')} (ID: {item.get('TOKEN_ID')})")
    
    # Test with offset parameter
    print("\nTest 2: Using offset parameter")
    print("Fetching with offset 0, limit 10...")
    offset0_response = await fetch_from_api("tokens", {"limit": limit, "offset": 0})
    offset0_items = offset0_response.get("data", [])
    
    print("Fetching with offset 10, limit 10...")
    offset10_response = await fetch_from_api("tokens", {"limit": limit, "offset": limit})
    offset10_items = offset10_response.get("data", [])
    
    # Print first 3 tokens from each offset for comparison
    print("\nFirst 3 tokens from offset 0:")
    for i, item in enumerate(offset0_items[:3]):
        print(f"{i+1}. {item.get('TOKEN_SYMBOL')} (ID: {item.get('TOKEN_ID')})")
    
    print("\nFirst 3 tokens from offset 10:")
    for i, item in enumerate(offset10_items[:3]):
        print(f"{i+1}. {item.get('TOKEN_SYMBOL')} (ID: {item.get('TOKEN_ID')})")
    
    # Check for pagination type
    if set(item.get('TOKEN_ID') for item in page0_items) == set(item.get('TOKEN_ID') for item in offset0_items):
        print("\nPage 0 matches Offset 0 results ✓")
    else:
        print("\nPage 0 differs from Offset 0 results ✗")
    
    # Compare page 1 with offset 10
    if set(item.get('TOKEN_ID') for item in page1_items) == set(item.get('TOKEN_ID') for item in offset10_items):
        print("Page 1 matches Offset 10 results - API likely uses offset-based pagination ✅")
    else:
        # Check if page 1 is same as page 0
        if set(item.get('TOKEN_ID') for item in page0_items) == set(item.get('TOKEN_ID') for item in page1_items):
            print("Page 0 and Page 1 return same data - pagination may be broken ⚠️")
        else:
            print("Page 1 differs from Offset 10 - inconclusive pagination method ❓")
    
    # Alternative tests with other possible parameter names
    print("\nTest 3: Trying other possible pagination parameters")
    # Try skip parameter
    try:
        print("Testing 'skip' parameter...")
        skip_response = await fetch_from_api("tokens", {"limit": limit, "skip": limit})
        skip_items = skip_response.get("data", [])
        if skip_items and skip_items != page0_items:
            print(f"'skip' parameter returned {len(skip_items)} different items - might be valid ✅")
        else:
            print("'skip' parameter doesn't seem to work as expected")
    except Exception as e:
        print(f"Error with 'skip' parameter: {e}")
    
    # Try start parameter
    try:
        print("\nTesting 'start' parameter...")
        start_response = await fetch_from_api("tokens", {"limit": limit, "start": limit})
        start_items = start_response.get("data", [])
        if start_items and start_items != page0_items:
            print(f"'start' parameter returned {len(start_items)} different items - might be valid ✅")
        else:
            print("'start' parameter doesn't seem to work as expected")
    except Exception as e:
        print(f"Error with 'start' parameter: {e}")
    
    print("\nTest complete!")
    
    # Conclusion
    print("\n====== CONCLUSION ======")
    print("Based on the tests, it appears that:")
    
    if set(item.get('TOKEN_ID') for item in page1_items) == set(item.get('TOKEN_ID') for item in offset10_items):
        print("✅ The API likely uses OFFSET-based pagination with the 'offset' parameter")
        print("   The 'page' parameter might be multiplied by 'limit' to calculate the offset")
    elif offset10_items and offset10_items != offset0_items:
        print("✅ The API supports OFFSET-based pagination with the 'offset' parameter")
        print("❓ The 'page' parameter may work differently than expected")
    elif set(item.get('TOKEN_ID') for item in page0_items) == set(item.get('TOKEN_ID') for item in page1_items):
        print("⚠️ The API pagination appears to be BROKEN - all pages return the same data")
    else:
        print("❓ The pagination method is INCONCLUSIVE - further investigation needed")

if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(test_offset_pagination()) 