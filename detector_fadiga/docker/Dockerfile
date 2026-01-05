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

# Expor porta única para HTTP + WebSocket
EXPOSE 8765

CMD ["python", "serve.py"]
