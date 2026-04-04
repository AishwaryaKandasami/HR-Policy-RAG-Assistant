# Dockerfile — HR Policy Q&A Bot Backend
# ============================================

# Step 1: Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Install system dependencies for build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file into the container
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the backend source code and demo docs
COPY . .
COPY ./demo_docs ./demo_docs

# Step 7: Expose the port that the app runs on (FastAPI default)
EXPOSE 8000

# Step 8: Set health check for Hugging Face Spaces
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Step 9: Start the FastAPI server using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
