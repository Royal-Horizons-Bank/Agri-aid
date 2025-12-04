FROM node:20-alpine AS builder
WORKDIR /app
RUN npm install -g http-server
COPY frontend /app
EXPOSE 3000
CMD ["http-server", "/app", "-p", "3000", "-c-1"]
