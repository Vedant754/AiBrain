import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Optional convenience: lets the frontend call "/api/..." without
    // hardcoding http://localhost:8000 everywhere. Real CORS is already
    // handled in the backend regardless.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
