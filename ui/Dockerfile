FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# --host so the dev server binds 0.0.0.0 -- Vite defaults to loopback, which
# inside a container is unreachable through the "5173:5173" port mapping.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
