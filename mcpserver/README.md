# LangConnect MCP Servers

This directory contains Model Context Protocol (MCP) server implementations for LangConnect.

## Servers

### stdio Server (`mcp_server.py`)

Standard MCP server using stdio transport. Used with Claude Desktop and other MCP clients.

```bash
uv run python mcpserver/mcp_server.py
```

### SSE Server (`mcp_sse_server.py`)

MCP server using SSE (Server-Sent Events) transport for web clients.

```bash
uv run python mcpserver/mcp_sse_server.py
```

## Configuration

### MCP Config (`mcp_config.json`)

For Claude Desktop, use the generated config:
```json
{
  "mcpServers": {
    "langconnect-rag-mcp": {
      "command": "/path/to/python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "API_BASE_URL": "http://localhost:8888"
      }
    }
  }
}
```

### Environment Variables

- `API_BASE_URL`: LangConnect API server URL (default: `http://localhost:8080`)
- `SSE_PORT`: Port for SSE server (default: `8765`)
- `OPENAI_API_KEY`: Required for multi-query generation

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python mcpserver/mcp_server.py
```

For the SSE server:
1. Start the server: `uv run python mcpserver/mcp_sse_server.py`
2. In MCP Inspector, connect to `http://localhost:8765` with SSE transport
