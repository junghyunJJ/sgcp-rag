#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LangConnect Installation Script ===${NC}"
echo ""

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}pnpm is not installed. Installing pnpm...${NC}"
    npm install -g pnpm
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}docker-compose is not installed. Please install docker-compose first.${NC}"
    exit 1
fi

# Change to Next.js frontend directory
echo -e "${GREEN}1. Installing frontend dependencies...${NC}"
cd next-connect-ui

# Install dependencies
pnpm install

# Build the Next.js app
echo -e "${GREEN}2. Building Next.js frontend...${NC}"
pnpm run build

# Return to root directory
cd ..

# Build Docker images
echo -e "${GREEN}3. Building Docker images...${NC}"
docker-compose build

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo -e "${YELLOW}To start the services, run:${NC}"
echo "  docker-compose up -d"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo "  docker-compose logs -f"
echo ""
echo -e "${YELLOW}Access the services at:${NC}"
echo "  - Frontend: http://localhost:3000"
echo "  - API: http://localhost:8080"
echo "  - API Docs: http://localhost:8080/docs"