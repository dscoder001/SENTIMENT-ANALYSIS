FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (code + dataset).
COPY . .

# Train the model at build time so the image ships with a ready-to-serve
# artifact, and the model is pickled with the exact library versions above.
RUN python train.py

# Configure Streamlit
RUN mkdir -p ~/.streamlit
COPY streamlit_config.toml ~/.streamlit/config.toml

# Streamlit's default port.
EXPOSE 8501

# Bind to 0.0.0.0 so the container is reachable from the host browser.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]
