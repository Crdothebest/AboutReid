import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    open: 'chrome', // 指定使用Chrome浏览器打开
    proxy: {
      '/api': 'http://localhost:8001',
      '/datasets': 'http://localhost:8001'
    }
  }
})
