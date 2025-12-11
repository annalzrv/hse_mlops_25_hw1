FROM python:3.12-slim

WORKDIR /app

# Create directories for logs
RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Mount points for input/output
VOLUME /app/input
VOLUME /app/output

CMD ["python", "./app/app.py"]
