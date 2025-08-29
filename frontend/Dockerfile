# Dev: Vite com hot reload
FROM node:20-alpine AS dev
WORKDIR /app
COPY docker-dev-entrypoint.sh /usr/local/bin/dev-entrypoint
RUN chmod +x /usr/local/bin/dev-entrypoint
EXPOSE 5173
CMD ["dev-entrypoint"]

# Build de produção
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci || npm install
COPY . .
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN echo "Baking VITE_API_URL=$VITE_API_URL" && npm run build

# Nginx para servir estático
FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]