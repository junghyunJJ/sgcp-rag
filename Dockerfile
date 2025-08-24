# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    curl \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN uv venv && \
    uv pip install -r requirements.txt && \
    uv pip install "unstructured[docx]"

# Final stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    libxml2 \
    libxslt1.1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv in the final stage for runtime use
RUN pip install --no-cache-dir uv

# Create non-root user
RUN groupadd -r langconnect && useradd -r -g langconnect langconnect

WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /home/langconnect/.cache && \
    chown -R langconnect:langconnect /home/langconnect/.cache

# Copy virtual environment from builder
COPY --from=builder --chown=langconnect:langconnect /app/.venv /app/.venv

# Copy application code
COPY --chown=langconnect:langconnect . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER langconnect

# Expose ports for API and MCP server
EXPOSE 8080 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()" || exit 1

# Default command runs the API server
CMD ["uvicorn", "langconnect.server:APP", "--host", "0.0.0.0", "--port", "8080"]
