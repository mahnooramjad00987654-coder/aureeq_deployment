## 🐳 Shared Environments (Advanced)
If your VPS already has other containers running (e.g., Nginx Proxy Manager, Traefik, or other websites):

1.  **Avoid Port Conflicts**: In `docker-compose.vps.yml`, if port `80` or `443` is already taken, you can either:
    *   Stop the existing service temporarily.
    *   OR (Recommended) Disable the `aureeq-proxy` service in the compose file and point your existing external proxy to `aureeq-frontend:80` and `aureeq-backend:8001`.
2.  **Isolated Network**: AUREEQ uses a private network (`aureeq-net`), so it will not interfere with your other containers' internal traffic.

## 🚀 Step-by-Step Launch (Standard)

**NOTE:** To avoid conflicts with existing websites on your VPS, AUREEQ now runs on port **8080** (HTTP) and **8443** (HTTPS).

1.  **Transfer Files**: Copy all files to your VPS folder.
2.  **Add Certificates**: Place `fullchain.pem` and `privkey.pem` in the `./certs` folder.
3.  **Set Environment Variables**: Create a `.env` file and add your `OPENAI_API_KEY`.
4.  **Launch Services**:
    ```bash
    docker-compose up -d --build
    ```
5.  **Access**: Open `http://your-vps-ip:8080` in your browser.

## 🛠 Troubleshooting
*   **"Bind: address already in use"**: This means something is already blocked on port 80/443. I have fixed this by moving AUREEQ to 8080/8443.
*   **GPU Error**: GPU reservations have been removed. AUREEQ will run on your CPU.
