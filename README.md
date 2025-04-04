# tokens-mcp

This project is an MCP (Model Control Protocol) server for token-metrics. It provides functionality for managing and analyzing token metrics through a standardized protocol interface.

## Installation

**Install uv** - Installing python uv package manager

First, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
uv sync
```

Most of the modern IDEs support uv out of the box, however in terminal you might want to activate virtual environment explcietly:
```
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Linux/macOS
```


**Install Node.js** - Using nvm
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
   sudo apt purge nodejs # in case you have outdated nodejs
   nvm install node
   npm install -g which cross-spawn
   ```
The reason why we need nodejs is because https://github.com/modelcontextprotocol/python-sdk library uses is under the hood for no goo reason.
python-sdk is a trash-quality library but because it is written by Anthropic everybody is using it and we also have to.

# Usage


## Running MCP server

starting the server
```bash
mcp run server.py
```


```bash
mcp dev server.py
```

Troubleshooting: in some cases it complains about which and cross-spawn, for this use-case
```
npx clear-npx-cache
npm install cross-spawn which --save-dev
```

Open http://127.0.0.1:6274 to inspect MCP server