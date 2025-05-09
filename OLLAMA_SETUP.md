# Ollama Integration Setup

This document explains how to set up and use Ollama models with the tRNA Ontology Retriever.

## Prerequisites

1. Install Ollama on your system:
   - Download from [ollama.com](https://ollama.com)
   - Follow installation instructions for your platform

2. Start the Ollama service:
   - After installation, Ollama should run as a service
   - By default, it runs on http://localhost:11434
   - Verify that Ollama is running by checking if the service is active

3. Pull the model you want to use:
   - Open a terminal
   - Run: `ollama pull <model_name>`
   - Example: `ollama pull cas/ministral-8b-instruct-2410_q4km` (default)
   - Other options: `ollama pull llama3` or `ollama pull mistral`
   - Wait for the download to complete

## Setting Up in the tRNA Ontology Retriever

### Option 1: Using the UI

1. Start the application:
   ```
   python gradio_ui.py
   ```

2. In the model dropdown, select `ollama:custom`

3. Click on "Advanced Options"

4. Set the following:
   - LLM Provider: Select "ollama"
   - Ollama URL: Default is "http://localhost:11434" (change if needed)
   - In some versions, you may need to specify the model name separately

5. Enter your query and click "Process Query"

### Option 2: Using Environment Variables

You can set these environment variables before starting the application:

```
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434  # Change if needed
```

Then start the application normally:
```
python gradio_ui.py
```

## Available Ollama Models

Default model:
- cas/ministral-8b-instruct-2410_q4km (default - recommended for this application)

Other common Ollama models you can use:

- llama3
- llama2
- mistral
- mixtral
- gemma:2b
- gemma:7b
- codegemma

To use these models, you need to:
1. Pull them using `ollama pull <model_name>`
2. When using `ollama:custom`, the model name specified in the "Advanced Options" should match the one you pulled

## Troubleshooting

1. **Model not found errors**:
   - Make sure you have pulled the model using `ollama pull <model_name>`
   - Check the model name is correct and matches exactly what you pulled
   - Run `ollama list` to see all available models

2. **Connection errors**:
   - Verify Ollama is running: `ps aux | grep ollama`
   - Check if you can access the Ollama API directly:
     ```
     curl -X POST http://localhost:11434/api/generate -d '{
       "model": "llama2",
       "prompt": "Hello, world!"
     }'
     ```

3. **Other issues**:
   - Check the application logs for detailed error messages
   - Restart Ollama if it's unresponsive: `ollama serve`

## Additional Tips

- The first request to a model might be slow as Ollama loads the model into memory
- Subsequent requests should be faster
- Ollama models run locally on your machine using its resources
- Model performance depends on your hardware (especially RAM and GPU)
- Some models require specific hardware (e.g., GPU) to run efficiently