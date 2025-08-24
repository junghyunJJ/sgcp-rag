# LangConnect MCP Servers

This directory contains Model Context Protocol (MCP) server implementations for LangConnect.

## Authentication

Both MCP servers use Supabase JWT authentication. You need to provide a valid access token to use these servers.

### How to Get Your Access Token

#### Option 1: Using the Helper Script (Recommended)

Run the provided helper script:
```bash
cd mcpserver
python get_access_token.py
```

Enter your email and password when prompted. The script will:
- Sign you in
- Test the token
- Display the access token for you to copy

#### Option 2: From the Next.js UI

1. **Sign in to the Next.js UI** at http://localhost:3000
2. **Open Developer Tools** in your browser (F12)
3. **Go to the Application/Storage tab**
4. **Find Session Storage** for localhost:3000
5. **Look for the `access_token` key**
6. **Copy the JWT token value**

### Using the Access Token

#### For Standard MCP Server (stdio)

Update `mcp_config.json`:
```json
{
  "mcpServers": {
    "langconnect-rag-mcp": {
      "command": "/path/to/python",
      "args": [
        "/path/to/mcp_server.py"
      ],
      "env": {
        "API_BASE_URL": "http://localhost:8080",
        "SUPABASE_JWT_SECRET": "YOUR_JWT_TOKEN_HERE"
      }
    }
  }
}
```

#### For SSE Server

Set the environment variable:
```bash
export SUPABASE_JWT_SECRET="YOUR_JWT_TOKEN_HERE"
docker compose up -d mcp-sse
```

Or add it to your `.env` file:
```
SUPABASE_JWT_SECRET=YOUR_JWT_TOKEN_HERE
```

## Token Expiration

Supabase JWT tokens expire after a certain period (typically 1 hour). When your token expires:

1. Sign in again through the Next.js UI
2. Get the new access token
3. Update your configuration with the new token

## Testing with MCP Inspector

To test the MCP server with [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

### Option 1: Using npx (Recommended)

```bash
# Set your access token
export SUPABASE_JWT_SECRET="your-jwt-token-here"

# Run with MCP Inspector
npx @modelcontextprotocol/inspector python mcp/mcp_server.py
```

This will start the MCP server and open the Inspector UI in your browser.

### Option 2: Using the SSE Server

1. Start the SSE server:
```bash
export SUPABASE_JWT_SECRET="your-jwt-token-here"
python mcp/mcp_langconnect_sse_server.py
```

2. In MCP Inspector:
   - URL: `http://localhost:8765`
   - Transport: `sse`

## Security Notes

- **Never commit tokens** to version control
- Keep your `.env` file in `.gitignore`
- Tokens are user-specific and grant access to that user's data
- Always use HTTPS in production environments