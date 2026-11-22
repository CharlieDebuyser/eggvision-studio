/// <reference types="vitest/config" />
import { defineConfig, loadEnv, type PluginOption } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { mockApiPlugin } from './mock/mockApi.ts';

export default defineConfig(({ mode }) => {
  const env = { ...process.env, ...loadEnv(mode, process.cwd(), '') };
  const useMock = env.VITE_MOCK === '1';
  // Cible du proxy /api (backend FastAPI) ; le port du dev server se règle avec `--port`.
  const apiTarget = env.VITE_API_TARGET || 'http://127.0.0.1:8765';
  const plugins: PluginOption[] = [react()];
  if (useMock) plugins.push(mockApiPlugin());

  return {
    base: './',
    plugins,
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    build: {
      outDir: '../src/eggvision/web',
      emptyOutDir: true,
      chunkSizeWarningLimit: 1500,
    },
    server: {
      port: 5173,
      strictPort: false,
      // En mode mock, le middleware répond à /api avant le proxy.
      proxy: useMock ? undefined : { '/api': { target: apiTarget, changeOrigin: false } },
    },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      css: false,
    },
  };
});
