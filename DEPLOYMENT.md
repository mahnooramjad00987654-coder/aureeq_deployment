## 🐳 Shared Environments (Advanced)
If your VPS already has other containers running (e.g., Nginx Proxy Manager, Traefik, or other websites):

1.  **Avoid Port Conflicts**: In `docker-compose.vps.yml`, if port `80` or `443` is already taken, you can either:
    *   Stop the existing service temporarily.
    *   OR (Recommended) Disable the `aureeq-proxy` service in the compose file and point your existing external proxy to `aureeq-frontend:80` and `aureeq-backend:8001`.
2.  **Isolated Network**: AUREEQ uses a private network (`aureeq-net`), so it will not interfere with your other containers' internal traffic.

## 🚀 Step-by-Step Launch (Standard)

1.  **Transfer Files**: Copy all files to your VPS folder.
2.  **Add Certificates**: Place `fullchain.pem` and `privkey.pem` in the `./certs` folder.
3.  **Set Environment Variables**: Create a `.env` file and add your `OPENAI_API_KEY`.
4.  **Launch Services**:
    ```bash
    docker-compose up -d --build
    ```

## 🛠 Troubleshooting
*   **GPU Error**: GPU reservations have been removed. AUREEQ will run on your CPU.
*   **Port Conflicts**: If port 80/443 is used, disable the `proxy` service in `docker-compose.yml` and use your own Nginx/Caddy.
