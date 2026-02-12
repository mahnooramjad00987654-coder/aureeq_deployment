# VPS Deployment Plan for AUREEQ

To make AUREEQ "Docker-ready" and self-contained for your VPS, we will:

1.  **Pre-package AI Models**: Create a custom Ollama image that includes `llama3.2:1b` and `nomic-embed-text`. This ensures the container works offline and doesn't need to pull large models on startup.
2.  **HTTPS with Nginx**: Configure Nginx as a reverse proxy that uses your own SSL certificates (placed in a `certs/` volume).
3.  **Isolated Internal Network**: Use a dedicated Docker bridge network so it doesn't conflict with your other containers.
4.  **Persistence**: Use Docker volumes for the SQLite database and internal state.

## File Structure for Deployment
- `docker-compose.yml` - Orchestration.
- `Dockerfile.backend` - Optimized Python backend.
- `Dockerfile.frontend` - Production build of the React app.
- `Dockerfile.ollama` - Custom image with baked-in models.
- `nginx.conf` - Reverse proxy with SSL config.
- `certs/` - Folder where you will place your `fullchain.pem` and `privkey.pem`.

## Isolated Network Strategy
We will name the network `aureeq-net` and ensure container names are prefixed with `aureeq-`.
