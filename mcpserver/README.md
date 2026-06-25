# SGCP-RAG MCP Servers

This directory contains Model Context Protocol (MCP) server implementations for SGCP-RAG.

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
    "sgcp-rag": {
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

- `API_BASE_URL`: SGCP-RAG API server URL (default: `http://localhost:8080`)
- `SSE_PORT`: Port for SSE server (default: `8765`)
- `OLLAMA_BASE_URL`: Shared fallback Ollama endpoint for local LLMs (default: `http://localhost:5000`)
- `QUERY_EXPANSION_LLM_BASE_URL`: Query expansion Ollama endpoint; falls back to `OLLAMA_BASE_URL`
- `AGENT_LLM_BASE_URL`: Agentic RAG Ollama endpoint; falls back to `OLLAMA_BASE_URL`
- `QUERY_EXPANSION_LLM_PROVIDER`: Query expansion provider: `auto`, `ollama`, or `openai` (default: `auto`)
- `QUERY_EXPANSION_LLM_MODEL`: Ollama query expansion model (default: `qwen3.5:35b`)
- `QUERY_EXPANSION_OPENAI_MODEL`: OpenAI fallback model for query expansion (default: `gpt-5.4`)
- `AGENT_LLM_PROVIDER`: Agentic RAG provider: `auto`, `openai`, `google`, or `ollama`
- `AGENT_LLM_MODEL`: Agentic RAG Ollama model for `auto`/`ollama` (default: `qwen3.5:122b`)
- `AGENT_LLM_OPENAI_MODEL`: OpenAI fallback model for Agentic RAG `auto` mode (default: `gpt-5.4`)
- `AGENT_LLM_TEMPERATURE`: Agentic RAG LLM temperature (default: `0`)
- `SNI_LLM_PROVIDER`: SNI rebuild provider: `openai`, `google`, or `ollama`
- `SNI_LLM_BASE_URL`: SNI rebuild Ollama endpoint
- `SNI_LLM_MODEL`: SNI rebuild model name, for example `qwen3.5:397b-cloud`
- `SNI_LLM_TEMPERATURE`: SNI rebuild LLM temperature
- `OPENAI_API_KEY`: Required only when OpenAI is selected or used as fallback

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python mcpserver/mcp_server.py
```

For the SSE server:
1. Start the server: `uv run python mcpserver/mcp_sse_server.py`
2. In MCP Inspector, connect to `http://localhost:8765` with SSE transport
