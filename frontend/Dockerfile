# Use Node.js base image
FROM node:20-slim AS build

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY frontend/ .

# Build the application
RUN npm run build

# Use a lightweight server to serve the build artifacts
FROM node:20-slim

WORKDIR /app

# Install 'serve' package
RUN npm install -g serve

# Copy build artifacts from the build stage
COPY --from=build /app/dist /app/dist

# Expose port 5173 (standard for this project based on README)
EXPOSE 5173

# Serve the application
CMD ["serve", "-s", "dist", "-l", "5173"]
