# app.py
from mcp.server.fastmcp import FastMCP
from tmai_api import TokenMetricsClient
import os
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Create MCP server instance
mcp = FastMCP("Token Metrics Crypto Analytics")

# Configure official SDK client
client = TokenMetricsClient(api_key=os.getenv("TM_API_KEY"))

### Resources
@mcp.resource("coins/list")
async def get_coins() -> dict:
    """Get all tracked cryptocurrencies"""
    try:
        return client.coins.list().to_dict()
    except Exception as e:
        return {"error": str(e)}

@mcp.resource("coin/{symbol}")
async def get_coin(symbol: str) -> dict:
    """Get detailed coin information"""
    try:
        return client.coins.get(symbol=symbol).to_dict()
    except Exception as e:
        return {"error": str(e)}

@mcp.resource("market")
async def market_data() -> dict:
    """Current market overview"""
    try:
        return client.market.summary().to_dict()
    except Exception as e:
        return {"error": str(e)}

### Tools
@mcp.tool()
async def search_coins(query: str) -> dict:
    """Search cryptocurrencies by name/symbol"""
    try:
        return client.search(query).to_dict()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool() 
async def get_grades(symbol: str) -> dict:
    """Get investment grades for a coin"""
    try:
        return {
            "trader": client.grades.trader(symbol).to_dict(),
            "investor": client.grades.investor(symbol).to_dict()
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def price_prediction(symbol: str) -> dict:
    """Get AI price predictions"""
    try:
        return client.predictions.get(symbol).to_dict()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def technical_analysis(symbol: str) -> dict:
    """Technical indicators analysis"""
    try:
        return client.analysis.technical(symbol).to_dict()
    except Exception as e:
        return {"error": str(e)}

### Prompts
@mcp.prompt()
def analyze_coin(symbol: str) -> str:
    return f"""Analyze {symbol} using Token Metrics data:
1. Current market position
2. Investment grades
3. Price predictions
4. Technical indicators
5. Buy/sell recommendation"""

@mcp.prompt()
def market_report() -> str:
    return """Provide comprehensive market report:
- Top 5 coins by market cap
- Notable price movements
- Emerging trends
- Recommended portfolio adjustments"""

if __name__ == "__main__":
    if not os.getenv("TM_API_KEY"):
        print("Error: TM_API_KEY missing in .env file")
        print("Create .env with: TM_API_KEY=your_api_key_here")
        exit(1)
        
    mcp.run()
