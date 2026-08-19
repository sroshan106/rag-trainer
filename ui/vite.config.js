import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev server proxies /api to the FastAPI backend so the browser never talks
// cross-origin -- avoids depending on the CORS allowlist in src/api/app.py
// staying in sync with whatever port `vite dev` happens to pick.
// VITE_API_PROXY_TARGET overrides the target for docker compose, where the
// backend is reachable at the "app" service name, not 127.0.0.1.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
