"""
Main entry point for the Token Metrics MCP server.
"""
import os
import asyncio
from typing import Literal
import typer
from dotenv import load_dotenv
from tokens_mcp.server import mcp

app = typer.Typer()

@app.command()
def run_server(transport: Literal["stdio", "sse"] = "stdio"):
    """
    Run the Token Metrics MCP server.
    
    Args:
        transport: The transport protocol to use, either "stdio" or "sse"
    """
    # Load environment variables
    load_dotenv(override=True)
    
    # Start the server
    print(f"Starting Token Metrics MCP server with {transport} transport...")
    mcp.run(transport=transport)

if __name__ == "__main__":
    app() 