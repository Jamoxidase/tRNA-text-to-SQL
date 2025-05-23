# tRNA Ontology Retriever

A natural language interface for querying tRNA databases using LLMs. This system uses Voyage AI for embeddings and can utilize Ollama for local LLM inference.

## Docker Setup (for Server Deployment)

### Requirements

- Docker and Docker Compose installed
- At least 8GB RAM (16GB+ recommended)
- Sufficient disk space for models and data

### Running with Docker

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Customize the volume paths in `docker-compose.yml` if needed:
   ```yaml
   volumes:
     - /data/scratch/larbales/trna-data:/data  # Application data
     - /data/scratch/larbales/ollama-data:/root/.ollama  # Ollama models
   ```

3. Start the container:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at:
   ```
   http://<server-ip>:11860
   ```

5. The first startup may take several minutes as the Ollama model is downloaded.

### Docker Container Features

- **All-in-one container**: Runs both the tRNA Ontology Retriever and Ollama in a single container
- **Persistent storage**: Maintains model files and data across container restarts
- **Pre-configured model**: Uses `mistral-openorca` by default
- **Network access**: Binds to all interfaces (0.0.0.0), making it accessible via SSH tunneling

### SSH Tunnel Access

If you're running this on a remote server, you can access it via SSH tunneling:

```bash
ssh -L 11860:localhost:11860 username@remote-server
```

Then access the interface at http://localhost:11860 in your local browser.

### Troubleshooting Docker Setup

- **Container fails to start**: Check Docker logs with `docker logs trna-service`
- **Model not loading**: Ensure there's enough disk space and memory
- **Query times out**: The model (mistral-openorca) may need more time for the first query, try again
- **Cannot connect via SSH tunnel**: Ensure the application is binding to 0.0.0.0 and that your SSH tunnel command is correct

## Local Setup

### Requirements

- Python 3.6+
- Dependencies listed in requirements.txt
- Ollama installed separately (for local LLM support)
- Internet connectivity (required for Voyage AI embeddings)

### Local Setup Steps

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Ollama (for local LLM support):
   - See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for details

### Running the Application Locally

Run the application with:
```bash
python3 gradio_ui.py
```

### Configuration

You can create a `.env` file to set configuration options:
```
# LLM Provider Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# LiteLLM Configuration (if needed)
# LITELLM_API_KEY=your_api_key_here
# LITELLM_BASE_URL=your_base_url_here

# Database Configuration
TRNA_DB_PATH=trna_db_v01.db
```

## System Architecture

- **Embeddings**: Generated using the Voyage AI API (external service)
- **LLM Inference**: Can be performed via:
  - Remote LLMs via LiteLLM proxy (claude-3-5-sonnet, openai-gpt-4, etc.)
  - Local LLMs using Ollama (ollama:mistral-openorca)
- **Database**: Local SQLite database for tRNA data storage and querying

### Using Ollama Models

To use the Ollama-based model:
1. Select "ollama:mistral-openorca" from the model dropdown in the UI
2. The system will automatically route the request to the Ollama API instead of LiteLLM
3. The Docker container comes with Ollama pre-installed and the mistral-openorca model pre-loaded

## Performance Considerations

- The default Docker configuration allocates 60GB of memory and 24 CPU cores
- Adjust these values based on your server's available resources