# Tokens MCP

MCP server for Token Metrics API.

## Project Structure

The project is organized using a standard Python package structure:

```
tokens-mcp/
├── src/                      # Source code directory
│   └── tokens_mcp/           # Main package
│       ├── __init__.py       # Package initialization
│       ├── main.py           # Entry point
│       ├── server.py         # MCP server implementation
│       ├── models.py         # Pydantic models
│       ├── helpers.py        # Helper functions
│       └── chart_utils.py    # Chart utilities
├── tests/                    # Test directory
│   ├── test_token_symbols.py # Token symbol tests
│   ├── test_pagination.py    # Pagination tests
│   └── ...                   # Other tests
├── launch_server.py          # Server launcher script
├── run_tests.py              # Test runner script
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -e .
```

### Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Server

```bash
./launch_server.py
```

You can inspect the server by running
```bash
uv run mcp dev
```

### Running Tests

```bash
pytest -v tests/
```

## Development

To set up a development environment:

```bash
pip install -e ".[dev]"
```
