FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Impedir prompts interativos durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Garantir que 'python' e 'pip' apontem para o 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Atualizar o pip e instalar setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install python dependencies usando o módulo do python para evitar mismatch
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create input/output directories
RUN mkdir -p input output

# Command to run the application
CMD ["python", "main.py"]
