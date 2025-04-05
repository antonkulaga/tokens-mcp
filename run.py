from eliot import start_action
from pycomfort.logging import to_nice_file, to_nice_stdout
from tokens_mcp.server import mcp
from pathlib import Path
from datetime import datetime

def main():
    """Main function to run when the server starts"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create file naming pattern with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_json = logs_dir / f"server_{timestamp}.json"
    log_file = logs_dir / f"server_{timestamp}.log"
    
    # there are some issues how mcp servers deal with log files. So far it does not output properly
    # note: so far you have to run the server with `uv run mcp run run.py`
    
    to_nice_file(log_json, log_file) 
    with start_action(action_type="main") as action:
        action.log(message_type="server_start", message="Starting MCP server...")
        mcp.run()

if __name__ == "__main__":
    main()
