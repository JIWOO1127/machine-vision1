import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Cloudflare Quick Tunnel은 실행할 때마다 임시 trycloudflare.com 도메인을 만든다.
    allowedHosts: ['.trycloudflare.com'],
    proxy: {
      '/api': 'http://127.0.0.1:5000',
    },
  },
})
