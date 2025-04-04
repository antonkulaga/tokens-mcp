import httpx
import os
import json
import asyncio
from dotenv import load_dotenv
from models import TokenInfo 
from pydantic import TypeAdapter

load_dotenv(override=True)

async def inspect_token_response():
    """Fetch tokens to inspect their structure and validate Pydantic models"""
    url = "https://api.tokenmetrics.com/v2/tokens"
    headers = {
        "accept": "application/json",
        "api_key": os.getenv("TOKEN_METRICS_API_KEY")
    }
    params = {"limit": 5}  # Get 5 tokens for better sampling
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            print("=== SAMPLE TOKEN RECORD ===")
            sample_token = data["data"][0]
            print(json.dumps(sample_token, indent=2))
            
            # Show all keys in the record
            print("\n=== ALL FIELDS ===")
            all_fields = set()
            
            # Collect fields from all tokens
            for token in data["data"]:
                all_fields.update(token.keys())
            
            # Print fields sorted alphabetically
            for key in sorted(all_fields):
                print(f"- {key}")
            
            # Examine if our model can handle these tokens
            print("\n=== VALIDATING MODELS ===")
            
            # Test validation with samples
            for i, token in enumerate(data["data"]):
                try:
                    # Use model_validate since we're using Pydantic v2
                    model = TokenInfo.model_validate(token)
                    print(f"✅ Token {i+1} validated successfully")
                    
                    # Check for extra fields
                    model_dict = model.model_dump()
                    model_fields = set(model_dict.keys())
                    token_fields = set(token.keys())
                    extra_fields = token_fields - model_fields
                    
                    if extra_fields:
                        print(f"   Extra fields detected: {', '.join(extra_fields)}")
                        
                except Exception as e:
                    print(f"❌ Token {i+1} validation failed: {e}")
            
            # Try to validate all tokens in one go
            try:
                token_adapter = TypeAdapter(list[TokenInfo])
                all_tokens = token_adapter.validate_python(data["data"])
                print(f"\n✅ Successfully validated all {len(all_tokens)} tokens")
            except Exception as e:
                print(f"\n❌ Batch validation failed: {e}")
        else:
            print("No data returned from API")

if __name__ == "__main__":
    asyncio.run(inspect_token_response()) 