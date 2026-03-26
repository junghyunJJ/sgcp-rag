.PHONY: build up down restart mcp test

build:
	@echo "🔨 Building Next.js application..."
	@cd next-connect-ui && npm install && npm run build
	@echo "✅ Next.js build completed!"
	@echo ""
	@echo "🔨 Building Docker images..."
	@docker-compose build
	@echo "✅ Docker build completed successfully!"
	@echo "📌 Run 'make up' to start the server"

up:
	@echo "🚀 Starting LangConnect server..."
	@docker-compose up -d
	@echo "✅ Server started successfully!"
	@echo "📌 Access points:"
	@echo "   - API Server: http://localhost:8888"
	@echo "   - API Docs: http://localhost:8888/docs"
	@echo "   - Next.js UI: http://localhost:3005"
	@echo "   - PostgreSQL: localhost:5432"

down:
	@echo "🛑 Stopping LangConnect server..."
	@docker-compose down
	@echo "✅ Server stopped successfully!"

restart:
	@echo "🔄 Restarting LangConnect server..."
	@docker-compose down
	@docker-compose up -d
	@echo "✅ Server restarted successfully!"

mcp:
	@echo "🔧 MCP configuration (mcpserver/mcp_config.json):"
	@cat mcpserver/mcp_config.json
	@echo ""
	@echo "✅ Copy the above JSON into your MCP client settings."

TEST_FILE ?= tests/unit_tests

test:
	./run_tests.sh $(TEST_FILE)

