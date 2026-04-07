# Dockerfile — HR Policy Q&A Bot Backend
# ============================================

# Step 1: Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Step 2: Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Step 3: Install system dependencies for build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Create a non-root user (Hugging Face Spaces recommendation)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Step 5: Set the working directory
WORKDIR $HOME/app

# Step 6: Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Step 7: Copy the backend source code and demo docs
COPY --chown=user . .
# demo_docs is usually already in the current directory, but let's be explicit
COPY --chown=user ./demo_docs ./demo_docs

# Step 8: Expose the port (FastAPI default for HF is 7860)
EXPOSE 7860

# Step 9: Set health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:7860/health || exit 1

# Step 10: Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

