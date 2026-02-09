#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
pid=$!

sleep 5

echo "🔴 Retrieving models..."
ollama pull phi3
ollama pull llama3.2:1b
ollama pull nomic-embed-text
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
