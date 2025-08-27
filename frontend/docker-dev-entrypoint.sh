#!/bin/sh
set -e

cd /app

# Instala dependências se o volume node_modules estiver vazio
if [ ! -d node_modules ] || [ -z "$(ls -A node_modules 2>/dev/null)" ]; then
  echo "[frontend] Instalando dependências..."
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
fi

echo "[frontend] Iniciando Vite (dev server)…"
exec npm run dev -- --host