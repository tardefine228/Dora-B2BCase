import { defineConfig } from 'vite';

// The UI is served by the FastAPI app under /ui/ (api/app/main.py), both in
// `npm run dev` and in the Docker image. Built asset URLs must match that prefix.
//
// In dev the browser talks to the Vite server only; /classify and /source are
// proxied to the API (default http://127.0.0.1:8000, override with API_PROXY_TARGET).
// The preview sandbox serves the page from *.e2b.app, so that host is allowed.
const apiTarget = process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  base: '/ui/',
  server: {
    host: '0.0.0.0',
    allowedHosts: ['.e2b.app'],
    proxy: {
      '/classify': { target: apiTarget, changeOrigin: true },
      '/source': { target: apiTarget, changeOrigin: true },
    },
  },
});
