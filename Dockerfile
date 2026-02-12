# Multi-stage build for minimal production image
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python packages with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.0 torchvision==0.17.0 && \
    pip install --no-cache-dir -r requirements.txt

# Final stage - minimal runtime
FROM python:3.12-slim

# Install only essential runtime libraries for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY app.py model.py ./
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/

# Create directories for runtime data (empty, will be populated at runtime)
RUN mkdir -p uploaded_videos instance

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
