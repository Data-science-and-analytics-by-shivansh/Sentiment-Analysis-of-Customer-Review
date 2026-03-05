# Multi-stage Dockerfile for Sentiment Analysis System
# Optimized for production deployment

# Stage 1: Builder
FROM python:3.9-slim as builder

LABEL maintainer="Data Science Team"
LABEL description="Enterprise Sentiment Analysis System"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_sentiment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements_sentiment.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY sentiment_analysis.py .
COPY config.py ./config.py 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/powerbi_export

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    DATA_PATH=/app/data \
    LOG_LEVEL=INFO

# Expose port for API (if running Flask)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sentiment_analysis; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "sentiment_analysis.py"]
