# Base image
FROM node:20-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 5173

# Command to run development server
# Host 0.0.0.0 is required for Docker access
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
