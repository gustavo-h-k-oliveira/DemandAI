import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 0.0.0.0 dentro do container
    port: 5173
    // Se preferir evitar CORS em dev, você pode proxyar:
    // proxy: { "/api": "http://backend:8000" }
  }
});