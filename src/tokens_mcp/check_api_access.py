import httpx
import asyncio
from typing import Dict, Any, List

# Base URL for Token Metrics API
BASE_URL = "https://api.tokenmetrics.com/v2"

async def check_endpoint(endpoint: str) -> Dict[str, Any]:
    """
    Check if an endpoint is accessible without API key
    
    Args:
        endpoint: API endpoint path
    
    Returns:
        Dict with status and response
    """
    url = f"{BASE_URL}/{endpoint}"
    headers = {
        "accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    result = {"endpoint": endpoint, "status": "unknown"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            result["status_code"] = response.status_code
            result["headers"] = dict(response.headers)
            
            try:
                result["data"] = response.json()
            except:
                result["data"] = response.text[:1000]  # Limit text to first 1000 chars
            
            if response.status_code == 200:
                result["status"] = "success"
                # Count items if we got a successful response with data
                if isinstance(result["data"], dict) and "data" in result["data"]:
                    if isinstance(result["data"]["data"], list):
                        result["item_count"] = len(result["data"]["data"])
            else:
                result["status"] = "error"
                
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

async def main():
    """Test access to all public endpoints"""
    print(f"Testing access to Token Metrics API endpoints that don't require API key...")
    
    # List of endpoints to check (focus on working ones)
    endpoints = [
        "ai-reports-tokens",
        "blockchains",
        "categories",
        "exchanges",
        "technical-indicators"
    ]
    
    # Check each endpoint
    results = []
    for endpoint in endpoints:
        print(f"Testing endpoint: {endpoint}...")
        result = await check_endpoint(endpoint)
        results.append(result)
        
        # Print result
        status = result["status_code"] if "status_code" in result else "Error"
        if status == 200:
            item_count = result.get("item_count", "unknown")
            print(f"✅ {endpoint}: Success (Status: {status}, Items: {item_count})")
        else:
            print(f"❌ {endpoint}: Failed (Status: {status})")
            if "data" in result:
                # Check data type before slicing
                response_data = result["data"]
                if isinstance(response_data, str):
                    print(f"   Response: {response_data[:200]}...")
                else:
                    print(f"   Response: {str(response_data)[:200]}...")
        
        print()  # Empty line between results
    
    # Print summary
    success_count = sum(1 for r in results if r.get("status_code") == 200)
    print(f"Summary: {success_count}/{len(results)} endpoints accessible")
    
    if success_count == 0:
        print("\n⚠️ All endpoints failed. Possible issues:")
        print("  - Network connectivity problems")
        print("  - API service is down")
        print("  - Cloudflare blocking access")
    elif success_count < len(results):
        print("\n⚠️ Some endpoints failed. Check individual results.")
    else:
        print("\n✅ All endpoints accessible without an API key!")
        
    # Print notice about broken endpoints
    print("\nNOTE:")
    print("- The 'coins' endpoint returns a 500 error but may work in the future")
    print("- The 'price' endpoint requires an API key")

if __name__ == "__main__":
    asyncio.run(main()) 