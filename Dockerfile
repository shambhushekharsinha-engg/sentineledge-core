# Intel Physical AI Challenge - Bimanual VLA Dockerfile
# Provides a 1-click reproducible environment for Judges

FROM ubuntu:22.04

# Setup environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git curl libgl1-mesa-glx libglib2.0-0 \
    libosmesa6-dev libglew-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . /workspace/

# Expose Gradio Web UI port
EXPOSE 7860

# Default command launches the interactive Web UI
CMD ["python3", "app.py"]
