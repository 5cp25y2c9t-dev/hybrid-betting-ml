FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data logs pretrained

# Expose Streamlit port
EXPOSE 8501

# Start monitor in background + Streamlit
CMD python real_time_monitor.py & streamlit run streamlit_dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
