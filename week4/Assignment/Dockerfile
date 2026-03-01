#parent image [cite: 39]
FROM python:3.11-slim

WORKDIR /Assignment

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI (if using a server) or standard port
EXPOSE 8000

# Execute the agent logic [cite: 39]
CMD ["python", "retailreturns.py"]