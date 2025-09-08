# Use a base image that includes Node.js
FROM node:22-alpine

# Install Python and pip
RUN apk add --no-cache python3 py3-pip

# Optional: Set an alias for python3 to just 'python'
RUN ln -sf python3 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy your application files (e.g., package.json, requirements.txt)
# COPY package.json ./
# COPY requirements.txt ./

# Run any necessary install commands
# RUN npm install
# RUN pip install -r requirements.txt

# Define your default command
# CMD ["node", "app.js"]