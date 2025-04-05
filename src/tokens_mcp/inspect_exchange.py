import httpx
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

async def inspect_exchange_structure():
    """Fetch token data to inspect the exchange list structure"""
    url = "https://api.tokenmetrics.com/v2/tokens"
    headers = {
        "accept": "application/json",
        "api_key": os.getenv("TOKEN_METRICS_API_KEY")
    }
    params = {"limit": 10}  # Fetch 10 tokens to find one with exchange data
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        data = response.json()
        
        # Find a token with exchange data
        for token in data.get("data", []):
            if token.get("EXCHANGE_LIST") and len(token["EXCHANGE_LIST"]) > 0:
                print(f"Found token {token['TOKEN_SYMBOL']} with exchange data")
                print("\n=== EXCHANGE_LIST STRUCTURE ===")
                print(json.dumps(token["EXCHANGE_LIST"][0], indent=2))
                break
                
        # Find a token with category data
        for token in data.get("data", []):
            if token.get("CATEGORY_LIST") and len(token["CATEGORY_LIST"]) > 0:
                print(f"\nFound token {token['TOKEN_SYMBOL']} with category data")
                print("\n=== CATEGORY_LIST STRUCTURE ===")
                print(json.dumps(token["CATEGORY_LIST"][0], indent=2))
                break

if __name__ == "__main__":
    asyncio.run(inspect_exchange_structure()) 