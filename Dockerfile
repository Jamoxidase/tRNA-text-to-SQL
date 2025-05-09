FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py .
COPY *.json .
COPY *.md .
COPY *.db* ./
COPY .env ./

# Expose the port for Gradio UI
EXPOSE 7860

# Create directory for database
RUN mkdir -p /data
VOLUME /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=ollama
ENV OLLAMA_BASE_URL=http://localhost:11434

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do\n\
  sleep 1\n\
done\n\
\n\
# Pull the model if not already downloaded\n\
ollama pull cas/ministral-8b-instruct-2410_q4km\n\
\n\
# Start the application (binding to all interfaces configured in the code)\n\
python gradio_ui.py\n\
\n\
# If Python app exits, kill Ollama\n\
kill $OLLAMA_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]