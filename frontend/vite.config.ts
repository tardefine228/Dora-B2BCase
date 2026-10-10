import { defineConfig } from 'vite';

// The UI is served by the FastAPI app under /ui/ (api/app/main.py), both in
// `npm run dev` and in the Docker image. Built asset URLs must match that prefix.
export default defineConfig({
  base: '/ui/',
});
