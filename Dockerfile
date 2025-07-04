# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY ./app ./app
COPY ./model ./model
COPY ./requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the model
RUN apt-get update && apt-get install -y curl && \
    mkdir -p model && \
    curl -L -o model/model.pkl "https://drive.google.com/uc?export=download&id=1owtzeQ_prCrWpaXunjBJFar5fGhaNY9O"

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
