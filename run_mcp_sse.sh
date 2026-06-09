#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          SGCP-RAG MCP SSE Server Launcher            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ Error: 'uv' is not installed.${NC}"
    echo -e "${YELLOW}Please install uv first: https://github.com/astral-sh/uv${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found.${NC}"
    echo -e "${YELLOW}Please copy .env.example to .env and configure it.${NC}"
    exit 1
fi

# Check if docker is running
if ! docker ps &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Docker doesn't seem to be running.${NC}"
    echo -e "${YELLOW}Make sure your API server is accessible at the configured URL.${NC}"
    echo ""
fi

# Check if API is accessible
API_URL=$(grep "API_BASE_URL" .env | cut -d '=' -f2 | tr -d '"' | tr -d ' ')
if [ -z "$API_URL" ]; then
    API_URL="http://localhost:8080"
fi

echo -e "${GREEN}🔍 Checking API server at $API_URL...${NC}"
if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API server is running!${NC}"
else
    echo -e "${RED}❌ API server is not accessible at $API_URL${NC}"
    echo -e "${YELLOW}Please make sure to run: docker compose up -d${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 Starting MCP SSE Server...${NC}"
echo ""

# Run the MCP SSE server
uv run python mcpserver/mcp_sse_server.py
