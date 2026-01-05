FROM python:3.11-slim

# Evita perguntas interativas ao instalar pacotes
ENV DEBIAN_FRONTEND=noninteractive

# Instala libs necessárias para OpenCV e MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expor portas: WebSocket (8765) e HTTP Health Check (8000)
EXPOSE 8000 8765

CMD ["python", "serve.py"]
